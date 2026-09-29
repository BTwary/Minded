"""Server-Side Request Forgery (SSRF) validation and protection layer for AA-OS.

Protects cloud storage endpoints, custom AI providers, and webhooks against:
- Private RFC 1918 subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- Loopback addresses (127.0.0.0/8, ::1) unless explicitly permitted for local Ollama
- Cloud instance metadata services (169.254.169.254, metadata.google.internal)
- Link-local and multicast addresses (fe80::/10, 224.0.0.0/4)

`validate_network_endpoint` / `assert_safe_endpoint` are pre-flight checks: safe to
use when validating user input at save-time (e.g. a settings form), but if a caller
validates a URL once and only connects to it later -- possibly on a different
request, minutes or days afterward -- a validated hostname can be silently
repointed at an internal address in between (DNS rebinding) with nothing to catch
it, since the eventual HTTP client re-resolves the hostname itself.

`resolve_pinned_target` closes that gap for the actual outbound call: it resolves
and validates the hostname once, then returns a target whose connection is pinned
to the exact validated IP, so there is no separate later resolution step for an
attacker to race. Callers that make the real network request should use this
immediately before connecting, not `assert_safe_endpoint` alone.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse


class SSRFSecurityError(ValueError):
    """Raised when a user-supplied network endpoint targets prohibited internal or cloud metadata infrastructure."""
    pass


FORBIDDEN_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.internal",
    "instance-data",
    "169.254.169.254",
    "169.254.170.2",  # AWS ECS task metadata
    "fd00:ec2::254",  # AWS IPv6 metadata
}


def validate_network_endpoint(url: Optional[str], allow_loopback: bool = False) -> Tuple[bool, str]:
    """Validate whether an external HTTP/HTTPS endpoint is safe to query.

    Args:
        url: Full URL string to validate.
        allow_loopback: If True, loopback addresses (127.0.0.1, ::1, localhost) are permitted.
                       Intended only for local self-hosted AI servers (e.g. Ollama).

    Returns:
        (True, "OK") if safe.
        (False, error_reason) if unsafe or invalid.
    """
    if not url or not url.strip():
        return True, "No URL provided."

    cleaned = url.strip()
    try:
        parsed = urlparse(cleaned)
    except Exception as exc:
        return False, f"Invalid URL structure: {exc}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"Prohibited URL scheme '{scheme}'. Only http and https are allowed."

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "URL does not specify a valid hostname."

    # Immediate check for prohibited cloud metadata hostnames
    if hostname in FORBIDDEN_HOSTNAMES:
        return False, f"Access to cloud instance metadata endpoint '{hostname}' is strictly forbidden."

    # Check loopback hostname explicitly
    if hostname in ("localhost", "localhost.localdomain"):
        if allow_loopback or os.getenv("AAOS_ALLOW_LOOPBACK_AI", "false").lower() in ("1", "true", "yes"):
            return True, "OK (Loopback permitted)"
        return False, "Access to localhost/loopback addresses is prohibited for cloud and external providers."

    # Check if hostname itself is directly an IP literal
    try:
        ip_obj = ipaddress.ip_address(hostname)
        return _check_ip_safety(ip_obj, allow_loopback)
    except ValueError:
        # Not a raw IP literal, proceed to DNS resolution
        pass

    # Resolve all DNS A/AAAA records to prevent DNS rebinding
    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return False, f"Could not resolve hostname '{hostname}': {exc}"
    except Exception as exc:
        return False, f"DNS resolution error for '{hostname}': {exc}"

    if not addr_info:
        return False, f"No IP addresses resolved for hostname '{hostname}'."

    for item in addr_info:
        sockaddr = item[4]
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"Invalid resolved IP '{ip_str}'."

        is_safe, reason = _check_ip_safety(ip_obj, allow_loopback)
        if not is_safe:
            return False, f"Resolved IP '{ip_str}' for hostname '{hostname}' violates security policy: {reason}"

    return True, "OK"


def _check_ip_safety(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, allow_loopback: bool) -> Tuple[bool, str]:
    """Inspect IP address properties against forbidden ranges."""
    # IPv4 mapped IPv6 addresses (e.g., ::ffff:127.0.0.1 or ::ffff:169.254.169.254)
    # and RFC 6052 / RFC 8215 NAT64 translated IPv4 addresses (64:ff9b::/96, 64:ff9b:1::/48)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return _check_ip_safety(ip.ipv4_mapped, allow_loopback)
        if ip in ipaddress.IPv6Network("64:ff9b::/96") or ip in ipaddress.IPv6Network("64:ff9b:1::/48"):
            embedded_ipv4 = ipaddress.IPv4Address(ip.packed[-4:])
            return _check_ip_safety(embedded_ipv4, allow_loopback)

    # Loopback
    if ip.is_loopback:
        if allow_loopback or os.getenv("AAOS_ALLOW_LOOPBACK_AI", "false").lower() in ("1", "true", "yes"):
            return True, "OK (Loopback permitted)"
        return False, f"Loopback address '{ip}' is prohibited."

    # Cloud link-local metadata (169.254.0.0/16, fe80::/10)
    if ip.is_link_local:
        return False, f"Link-local address '{ip}' is prohibited (cloud metadata protection)."

    # Private RFC 1918 subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, fc00::/7)
    if ip.is_private:
        return False, f"Private network address '{ip}' is prohibited."

    # Multicast / Reserved / Unspecified
    if ip.is_multicast:
        return False, f"Multicast address '{ip}' is prohibited."
    if ip.is_reserved:
        return False, f"Reserved address '{ip}' is prohibited."
    if ip.is_unspecified:
        return False, f"Unspecified address '{ip}' is prohibited."

    return True, "OK"


def assert_safe_endpoint(url: Optional[str], allow_loopback: bool = False, label: str = "Endpoint") -> None:
    """Enforce endpoint safety, raising SSRFSecurityError if invalid."""
    if not url:
        return
    is_safe, reason = validate_network_endpoint(url, allow_loopback=allow_loopback)
    if not is_safe:
        raise SSRFSecurityError(f"{label} '{url}' rejected for security: {reason}")


@dataclass(frozen=True)
class PinnedTarget:
    """A validated, DNS-pinned request target.

    `url` connects directly to the exact IP address that was validated -- its
    host component is an IP literal, never a hostname -- so the HTTP client
    cannot re-resolve the original hostname to something else after the check.
    `host_header` and `sni_hostname` carry the real hostname (and any explicit
    port) forward so the origin server's virtual-hosting and the TLS
    certificate check still see the name the caller asked for.
    """
    url: str
    host_header: str
    sni_hostname: str


def _first_safe_resolved_ip(
    hostname: str, port: int, allow_loopback: bool
) -> Tuple[Optional[ipaddress.IPv4Address | ipaddress.IPv6Address], str]:
    """Resolve `hostname` and return the first policy-safe IP found.

    Every resolved address is checked (not just the first), matching
    `validate_network_endpoint`'s strictness: if any resolved address is unsafe,
    resolution as a whole is rejected rather than risking a later connection
    happening to land on the unsafe one.
    """
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return None, f"Could not resolve hostname '{hostname}': {exc}"
    except Exception as exc:
        return None, f"DNS resolution error for '{hostname}': {exc}"

    if not addr_info:
        return None, f"No IP addresses resolved for hostname '{hostname}'."

    resolved_ips: list = []
    for item in addr_info:
        ip_str = item[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            return None, f"Invalid resolved IP '{ip_str}'."
        is_safe, reason = _check_ip_safety(ip_obj, allow_loopback)
        if not is_safe:
            return None, f"Resolved IP '{ip_str}' for hostname '{hostname}' violates security policy: {reason}"
        resolved_ips.append(ip_obj)

    return resolved_ips[0], "OK"


def resolve_pinned_target(url: Optional[str], allow_loopback: bool = False, label: str = "Endpoint") -> PinnedTarget:
    """Validate `url` and resolve it to a policy-safe, IP-pinned request target.

    Use this immediately before making the actual outbound HTTP request (not at
    settings-save time) so that validation and connection happen atomically --
    there is no window between check and use for a DNS record to change under
    the caller. Raises SSRFSecurityError if `url` is missing, malformed, or
    resolves to a prohibited address.
    """
    if not url or not url.strip():
        raise SSRFSecurityError(f"{label} is required and was not provided.")

    cleaned = url.strip()
    try:
        parsed = urlparse(cleaned)
    except Exception as exc:
        raise SSRFSecurityError(f"{label} '{url}' has an invalid URL structure: {exc}") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise SSRFSecurityError(f"{label} '{url}' uses prohibited scheme '{scheme}'. Only http and https are allowed.")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise SSRFSecurityError(f"{label} '{url}' does not specify a valid hostname.")

    if hostname in FORBIDDEN_HOSTNAMES:
        raise SSRFSecurityError(f"{label} '{url}' targets a forbidden cloud instance metadata endpoint '{hostname}'.")

    loopback_permitted = allow_loopback or os.getenv("AAOS_ALLOW_LOOPBACK_AI", "false").lower() in ("1", "true", "yes")

    port = parsed.port or (443 if scheme == "https" else 80)

    # Hostname is already a literal IP -- validate it directly, no resolution needed.
    try:
        literal_ip = ipaddress.ip_address(hostname)
        is_safe, reason = _check_ip_safety(literal_ip, loopback_permitted)
        if not is_safe:
            raise SSRFSecurityError(f"{label} '{url}' rejected for security: {reason}")
        pinned_ip = literal_ip
    except ValueError:
        # Not a literal IP.
        if hostname in ("localhost", "localhost.localdomain"):
            if not loopback_permitted:
                raise SSRFSecurityError(
                    f"{label} '{url}' rejected for security: access to localhost/loopback addresses "
                    "is prohibited for cloud and external providers."
                )
            pinned_ip = ipaddress.ip_address("127.0.0.1")
        else:
            pinned_ip, reason = _first_safe_resolved_ip(hostname, port, loopback_permitted)
            if pinned_ip is None:
                raise SSRFSecurityError(f"{label} '{url}' rejected for security: {reason}")

    if isinstance(pinned_ip, ipaddress.IPv6Address):
        pinned_host = f"[{pinned_ip}]"
    else:
        pinned_host = str(pinned_ip)
    pinned_netloc = f"{pinned_host}:{parsed.port}" if parsed.port else pinned_host

    pinned_url = urlunparse((parsed.scheme, pinned_netloc, parsed.path, parsed.params, parsed.query, ""))

    return PinnedTarget(
        url=pinned_url,
        host_header=parsed.netloc,
        sni_hostname=parsed.hostname,
    )
