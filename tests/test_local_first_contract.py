import os
from pathlib import Path


def test_local_first_paths_are_user_scoped(monkeypatch, tmp_path):
    from packages.analytics_core.src import platform_local_first as lf
    monkeypatch.setenv("AAOS_DATA_ROOT", str(tmp_path / "Minded" / "AAOS"))
    paths = lf.ensure_local_layout()
    assert paths["storage"].is_dir()
    assert paths["backups"].is_dir()
    assert paths["feedback"].is_dir()
    assert "Minded" in str(paths["root"])


def test_secret_key_persists_without_env(monkeypatch, tmp_path):
    from packages.analytics_core.src import platform_local_first as lf
    monkeypatch.setenv("AAOS_DATA_ROOT", str(tmp_path / "state"))
    monkeypatch.delenv("SECRET_KEY", raising=False)
    first = lf.persistent_secret_key()
    second = lf.persistent_secret_key()
    assert len(first) >= 32
    assert first == second


def test_local_first_defaults_disable_external_services(monkeypatch, tmp_path):
    from packages.analytics_core.src import platform_local_first as lf
    monkeypatch.setenv("AAOS_DATA_ROOT", str(tmp_path / "state"))
    defaults = lf.local_first_defaults()
    assert defaults["AI_ENABLED"] == "false"
    assert defaults["AI_PROVIDER"] == "none"
    assert defaults["STORAGE_PROVIDER"] == "local"
    assert defaults["TELEMETRY_ENABLED"] == "false"
    assert defaults["FEEDBACK_UPLOAD_ENABLED"] == "false"


def test_infrastructure_manager_defaults_to_local_and_no_ai(monkeypatch, tmp_path):
    monkeypatch.setenv("AAOS_DATA_ROOT", str(tmp_path / "state"))
    monkeypatch.delenv("STORAGE_PROVIDER", raising=False)
    monkeypatch.delenv("AI_ENABLED", raising=False)
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    from packages.analytics_core.src.providers.manager import InfrastructureManager
    InfrastructureManager._storage_provider = None
    InfrastructureManager._ai_provider = None
    storage = InfrastructureManager.get_storage_provider()
    ai = InfrastructureManager.get_ai_provider()
    assert storage.provider_type == "local"
    assert storage.is_local is True
    assert ai.provider_type == "none"
    assert ai.is_local is True
