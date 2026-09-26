"""Initialize a zero-friction local MindEd workspace.

Safe to run repeatedly. It does not contact the network, create an account,
or enable AI/cloud integrations.
"""
from __future__ import annotations
import os
from pathlib import Path

from packages.analytics_core.src.platform_local_first import ensure_local_layout, local_first_defaults


def main() -> None:
    paths = ensure_local_layout()
    defaults = local_first_defaults()
    print("Minded local workspace initialized")
    for name, value in paths.items():
        print(f"  {name}: {value}")
    print("\nDefault security/privacy posture:")
    for key in ("AI_ENABLED", "AI_PROVIDER", "STORAGE_PROVIDER", "TELEMETRY_ENABLED", "FEEDBACK_UPLOAD_ENABLED"):
        print(f"  {key}={defaults[key]}")
    print("  Network access is not required for deterministic analysis.")


if __name__ == "__main__":
    main()
