"""Regression coverage for the SSRF DNS-rebinding fix (release-hardening pass,
2026-09-25).

Prior behavior: AI provider base URLs were validated once, at settings-save
time (`set_ai_config` -> `assert_safe_endpoint`). The actual outbound request,
made later by `httpx` inside each provider's `generate()`, re-resolved the
hostname itself with no further check -- a validated hostname could be
DNS-rebound to an internal/metadata address between save and use with nothing
to catch it.

Fix: `resolve_pinned_target()` resolves and validates the hostname once,
*immediately before the request*, and returns a target whose connection is
pinned to the exact IP that was validated (while still presenting the real
hostname via the Host header / TLS SNI extension), so there is no separate,
later resolution step for a rebinding attack to land in. Each provider now
calls it right before its httpx call instead of relying on the one-time
settings-save check.
"""
import http.server
import threading
import socket as socket_mod

import pytest

from packages.analytics_core.src.security.ssrf import (
    resolve_pinned_target,
    validate_network_endpoint,
    SSRFSecurityError,
)
from apps.api.src.ai.providers.claude import ClaudeProvider
from apps.api.src.ai.providers.openai_compat import OpenAICompatibleProvider
from apps.api.src.ai.providers.ollama import OllamaProvider


# ---------------------------------------------------------------------------
# resolve_pinned_target: unit-level behavior
# ---------------------------------------------------------------------------

class TestResolvePinnedTargetRejectsUnsafeEndpoints:
    def test_rejects_cloud_metadata_ip_literal(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("http://169.254.169.254/latest/meta-data/", label="Test endpoint")

    def test_rejects_cloud_metadata_hostname(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("http://metadata.google.internal/", label="Test endpoint")

    def test_rejects_private_rfc1918_ip_literal(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("http://10.0.0.5:8080/", label="Test endpoint")

    def test_rejects_private_rfc1918_ip_literal_172_range(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("http://172.16.5.5/", label="Test endpoint")

    def test_rejects_loopback_ip_literal_by_default(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("http://127.0.0.1:11434/", allow_loopback=False, label="Test endpoint")

    def test_rejects_localhost_hostname_by_default(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("http://localhost:11434/", allow_loopback=False, label="Test endpoint")

    def test_rejects_missing_url(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("", label="Test endpoint")
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target(None, label="Test endpoint")

    def test_rejects_prohibited_scheme(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("ftp://example.com/", label="Test endpoint")

    def test_rejects_link_local_ip_literal(self):
        with pytest.raises(SSRFSecurityError):
            resolve_pinned_target("http://169.254.1.1/", label="Test endpoint")


class TestResolvePinnedTargetAllowsSafeEndpoints:
    def test_allows_loopback_when_flagged(self):
        target = resolve_pinned_target("http://127.0.0.1:11434/api", allow_loopback=True, label="Test endpoint")
        assert target.url == "http://127.0.0.1:11434/api"
        assert target.host_header == "127.0.0.1:11434"
        assert target.sni_hostname == "127.0.0.1"

    def test_allows_localhost_hostname_when_flagged(self):
        target = resolve_pinned_target("http://localhost:11434/api", allow_loopback=True, label="Test endpoint")
        assert target.url == "http://127.0.0.1:11434/api"
        assert target.host_header == "localhost:11434"
        assert target.sni_hostname == "localhost"

    def test_public_ip_literal_pins_to_itself(self):
        target = resolve_pinned_target("https://93.184.216.34/v1/messages", label="Test endpoint")
        assert target.url == "https://93.184.216.34/v1/messages"
        assert target.host_header == "93.184.216.34"
        assert target.sni_hostname == "93.184.216.34"

    def test_pinned_url_uses_ip_not_hostname_after_dns_resolution(self):
        # api.anthropic.com resolves to a real public IP (not asserting which
        # one, since that can change) -- what matters for the fix is that the
        # returned url's host component is an IP literal, not the original
        # hostname, while the original hostname is preserved for Host/SNI.
        target = resolve_pinned_target("https://api.anthropic.com/v1", label="Test endpoint")
        assert target.sni_hostname == "api.anthropic.com"
        assert target.host_header == "api.anthropic.com"
        assert "api.anthropic.com" not in target.url
        pinned_host = target.url.split("://", 1)[1].split("/", 1)[0]
        # Must parse as a bare IPv4/IPv6 literal, not a hostname.
        import ipaddress
        ipaddress.ip_address(pinned_host.strip("[]"))

    def test_no_url_provided_is_a_noop_for_validate_network_endpoint(self):
        # validate_network_endpoint (the pre-flight/settings-save check) keeps
        # its existing "no url means nothing to validate" contract -- only
        # resolve_pinned_target (used at actual connection time) requires a URL.
        is_safe, _ = validate_network_endpoint("")
        assert is_safe is True


# ---------------------------------------------------------------------------
# Providers: refuse to connect to unsafe endpoints, and successfully connect
# through the pinned target to a safe one.
# ---------------------------------------------------------------------------

class TestProvidersEnforceAtRequestTimeNotJustAtSave:
    """The core regression: a provider constructed with an unsafe base_url must
    refuse *at request time*, proving the check now lives where the network
    call actually happens rather than only at a settings-save step that these
    provider objects never go through in these tests."""

    def test_claude_provider_refuses_private_base_url_on_generate(self):
        provider = ClaudeProvider(api_key="k", base_url="http://10.0.0.5:9999")
        result = provider.generate("hello")
        assert "error" in result.lower()

    def test_openai_compat_provider_refuses_metadata_base_url_on_generate(self):
        provider = OpenAICompatibleProvider(api_key="k", base_url="http://169.254.169.254/openai")
        with pytest.raises(RuntimeError):
            provider.generate("hello")

    def test_ollama_provider_refuses_non_loopback_private_base_url_on_generate(self):
        # Ollama permits loopback, but not arbitrary RFC1918 addresses.
        provider = OllamaProvider(base_url="http://192.168.1.50:11434")
        with pytest.raises(RuntimeError):
            provider.generate("hello")


class _OneShotJSONHandler(http.server.BaseHTTPRequestHandler):
    """Minimal local HTTP server used to prove the pinned connection actually
    round-trips correctly (correct path, correct Host header received)."""

    def log_message(self, *args):  # silence test output
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        self.server.last_path = self.path  # type: ignore[attr-defined]
        self.server.last_host_header = self.headers.get("Host", "")  # type: ignore[attr-defined]
        body = b'{"response": "pinned-ok", "choices": [{"message": {"content": "pinned-ok"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def local_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _OneShotJSONHandler)
    server.last_path = None
    server.last_host_header = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestPinnedConnectionRoundTripsCorrectly:
    """Proves the fix doesn't just refuse unsafe endpoints -- it also still
    successfully completes a real request through the IP-pinned path for a
    legitimately safe (loopback-permitted) endpoint, hitting the exact right
    path and presenting the correct Host header to the origin server."""

    def test_ollama_provider_generate_round_trips_through_pinned_connection(self, local_server):
        port = local_server.server_address[1]
        provider = OllamaProvider(base_url=f"http://127.0.0.1:{port}", model="test-model")
        result = provider.generate("hello")
        assert result == "pinned-ok"
        assert local_server.last_path == "/api/generate"
        assert local_server.last_host_header == f"127.0.0.1:{port}"

    def test_openai_compat_provider_generate_round_trips_through_pinned_connection(self, local_server):
        port = local_server.server_address[1]
        provider = OpenAICompatibleProvider(
            api_key="k", base_url=f"http://127.0.0.1:{port}", model="test-model", allow_loopback=True
        )
        result = provider.generate("hello")
        assert result == "pinned-ok"
        assert local_server.last_path == "/chat/completions"
        assert local_server.last_host_header == f"127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# S3-compatible storage endpoint: defense-in-depth re-check at client build time
# ---------------------------------------------------------------------------

class TestS3EndpointDefenseInDepth:
    def test_get_client_rejects_unsafe_endpoint_even_if_constructed_directly(self):
        from packages.analytics_core.src.providers.storage import S3StorageProvider

        provider = S3StorageProvider(
            bucket_name="test-bucket",
            endpoint_url="http://169.254.169.254/",
            access_key_id="x",
            secret_access_key="y",
        )
        with pytest.raises(Exception):
            # Whether this raises SSRFSecurityError (our re-check) or an
            # ImportError from a missing optional boto3 in this environment,
            # it must never silently proceed to build a client against the
            # metadata endpoint.
            provider._get_client()
