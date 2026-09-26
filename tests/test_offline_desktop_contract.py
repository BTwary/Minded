from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_offline_entrypoint_has_loopback_and_network_guard() -> None:
    text = (ROOT / "packaging/windows/minded_entry.py").read_text(encoding="utf-8")
    assert 'AAOS_OFFLINE_MODE' in text
    assert 'socket.getaddrinfo = getaddrinfo' in text
    assert 'AAOS offline mode blocks non-loopback network access' in text


def test_offline_installer_has_no_package_manager_or_global_runtime_dependency() -> None:
    text = (ROOT / "packaging/windows/install.ps1").read_text(encoding="utf-8")
    assert 'pip install' not in text.lower().replace('node/npm/pip', '')
    assert 'npm ' not in text.lower()
    assert 'self-contained' in text.lower()
    assert 'MindedAAOS.exe' in text


def test_static_frontend_is_served_from_same_origin() -> None:
    api = (ROOT / "apps/web/src/lib/api.ts").read_text(encoding="utf-8")
    main = (ROOT / "apps/api/src/main.py").read_text(encoding="utf-8")
    config = (ROOT / "apps/web/next.config.js").read_text(encoding="utf-8")
    assert '|| "/api/v1"' in api
    assert 'StaticFiles(directory=_frontend_root, html=True)' in main
    assert 'output: "export"' in config


def test_offline_certification_is_fail_closed_for_missing_bundle() -> None:
    script = ROOT / "scripts/offline_desktop_certification.py"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert 'MindedAAOS.exe missing' in text
    assert 'OFFLINE DESKTOP CERTIFICATION' in text


def test_offline_certification_accepts_complete_fake_bundle(tmp_path) -> None:
    import json
    import subprocess
    import sys

    root = tmp_path / "bundle"
    app = root / "MindedAAOS"
    (app / "_internal").mkdir(parents=True)
    (app / "frontend").mkdir(parents=True)
    (app / "frontend" / "index.html").write_text("<html><body>AAOS</body></html>", encoding="utf-8")
    (app / "MindedAAOS.exe").write_bytes(b"stub")
    (app / "OFFLINE_MODE").write_text("1", encoding="ascii")
    (app / "MindedAAOS.exe.manifest.json").write_text(
        json.dumps({
            "offline": True,
            "network_policy": "loopback-only",
            "runtime_dependency": "self-contained",
        }),
        encoding="utf-8",
    )
    script = ROOT / "scripts/offline_desktop_certification.py"
    result = subprocess.run(
        [sys.executable, str(script), "--bundle", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_offline_certification_rejects_external_url_in_js_bundle(tmp_path) -> None:
    import json
    import subprocess
    import sys

    root = tmp_path / "bundle"
    app = root / "MindedAAOS"
    (app / "_internal").mkdir(parents=True)
    (app / "frontend").mkdir(parents=True)
    (app / "frontend" / "index.html").write_text("<html><body>AAOS</body></html>", encoding="utf-8")
    # Injected external URL into JavaScript bundle
    (app / "frontend" / "bundle.js").write_text('fetch("https://evil.example/x")', encoding="utf-8")
    (app / "MindedAAOS.exe").write_bytes(b"stub")
    (app / "OFFLINE_MODE").write_text("1", encoding="ascii")
    (app / "MindedAAOS.exe.manifest.json").write_text(
        json.dumps({
            "offline": True,
            "network_policy": "loopback-only",
            "runtime_dependency": "self-contained",
        }),
        encoding="utf-8",
    )
    script = ROOT / "scripts/offline_desktop_certification.py"
    result = subprocess.run(
        [sys.executable, str(script), "--bundle", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, "Must fail certification when external URLs are present in JS"
    assert "https://evil.example/x" in result.stdout or "external resource(s) found" in result.stdout

