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


def _make_bundle(tmp_path, *, wheels=("stub-1.0-py3-none-any.whl",), extra_js=None, index_html="<html><body>AAOS</body></html>"):
    import json

    root = tmp_path / "bundle"
    app = root / "MindedAAOS"
    (app / "_internal").mkdir(parents=True)
    (app / "frontend").mkdir(parents=True)
    (app / "frontend" / "index.html").write_text(index_html, encoding="utf-8")
    if extra_js is not None:
        (app / "frontend" / "bundle.js").write_text(extra_js, encoding="utf-8")
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
    if wheels:
        (app / "wheelhouse").mkdir()
        for name in wheels:
            (app / "wheelhouse" / name).write_bytes(b"PK")
    return root


def _certify(root):
    import subprocess
    import sys

    script = ROOT / "scripts/offline_desktop_certification.py"
    return subprocess.run(
        [sys.executable, str(script), "--bundle", str(root)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


def test_offline_certification_accepts_complete_fake_bundle(tmp_path) -> None:
    result = _certify(_make_bundle(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_offline_certification_rejects_external_url_in_js_bundle(tmp_path) -> None:
    root = _make_bundle(tmp_path, extra_js='fetch("https://evil.example/x")')
    result = _certify(root)
    assert result.returncode == 1, "Must fail certification when external URLs are present in JS"
    assert "https://evil.example/x" in result.stdout or "external resource(s) found" in result.stdout


def test_offline_certification_requires_non_empty_wheelhouse(tmp_path) -> None:
    missing = _certify(_make_bundle(tmp_path / "a", wheels=()))
    assert missing.returncode == 1, missing.stdout
    assert "[FAIL] local wheelhouse" in missing.stdout

    empty_root = _make_bundle(tmp_path / "b", wheels=())
    (empty_root / "MindedAAOS" / "wheelhouse").mkdir()
    empty = _certify(empty_root)
    assert empty.returncode == 1, empty.stdout
    assert "[FAIL] local wheelhouse" in empty.stdout


def test_url_allowlist_is_exact_host_not_prefix_or_substring() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("offline_cert", ROOT / "scripts/offline_desktop_certification.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for ok in (
        "http://127.0.0.1:8000/api",
        "http://localhost:3000/",
        "http://[::1]:8000/",
        "http://www.w3.org/2000/svg",
        "https://react.dev/errors/418",
        # inert literals Next.js 15.5.x really emits (whole-string match only)
        "http://n",
        "https://a/c%20d?a=1&c=3",
    ):
        assert mod._is_url_allowed(ok), ok
    for bad in (
        "https://a/evil",
        "https://a.evil.example",
        "http://n.evil.example",
        "http://n/../evil",
        "https://attacker.example",
        "http://x.evil.example/",
        "http://n.evil.example/",
        "https://evil.example/?next=localhost",
        "https://evil.example/127.0.0.1",
        "http://localhost.evil.example/",
        "https://github.com.evil.example/",
        "https://react.dev.evil.example/",
        "//evil.example/x.js",
    ):
        assert not mod._is_url_allowed(bad), bad


def test_offline_certification_rejects_attacker_hosts_that_match_old_prefixes(tmp_path) -> None:
    for url in ("https://attacker.example/x", "http://f.evil.example/y", "https://x.evil.example/z", "https://a.evil.example/q"):
        result = _certify(_make_bundle(tmp_path / url.split("//")[1].split("/")[0], extra_js=f'fetch("{url}")'))
        assert result.returncode == 1, url + "\n" + result.stdout


def test_offline_certification_rejects_external_html_src(tmp_path) -> None:
    root = _make_bundle(tmp_path, index_html='<script src="//cdn.evil.example/a.js"></script>')
    result = _certify(root)
    assert result.returncode == 1, result.stdout


# --- Windows installer / release-packaging contract (static; the C# cannot be compiled on Linux) ---

def test_windows_installer_version_is_not_hardcoded() -> None:
    setup = (ROOT / "packaging/windows/installer/MindedSetup.cs").read_text(encoding="utf-8")
    assert 'SetValue("DisplayVersion", ReleaseInfo.Version)' in setup
    assert '"30.0"' not in setup
    build = (ROOT / "packaging/windows/installer/build_installer.py").read_text(encoding="utf-8")
    assert "__version__" in build and "ReleaseVersion.g.cs" in build


def test_windows_uninstaller_stops_the_real_product_processes_and_fails_closed() -> None:
    text = (ROOT / "packaging/windows/installer/MindedUninstall.cs").read_text(encoding="utf-8")
    assert '"MindedAAOS"' in text and '"Minded"' in text
    assert "StopProductProcesses" in text
    # Must verify the processes actually exited and abort before deleting files if not.
    assert "WaitForExit" in text and "Nothing was removed" in text


def test_install_ps1_enforces_wheelhouse_like_the_certifier() -> None:
    text = (ROOT / "packaging/windows/install.ps1").read_text(encoding="utf-8")
    assert "wheelhouse" in text.lower() and "*.whl" in text


def test_package_zip_certification_profile_is_self_certifying() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("package_zip", ROOT / "scripts/package_zip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    required = (
        "tests/independent_release/manifest.toml",
        "data/certification/checksums.sha256",
        "real_stress_results_raw.json",
        "release_evidence/ATTESTATION.json",
    )
    for rel in required:
        assert not mod.is_excluded_path(rel, "certification"), rel
        assert mod.is_excluded_path(rel, "runtime"), rel
    assert mod.is_excluded_path("apps/web/node_modules/x.js", "certification")
    assert mod.is_excluded_path(".pytest_cache/v/cache", "certification")


def test_attestation_binds_windows_packaging_and_ships_gate_report(tmp_path, monkeypatch) -> None:
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("make_release_attestation", ROOT / "scripts/make_release_attestation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "packaging" in mod.SOURCE_DIRS
    assert ".exe" in mod.EXCLUDE_SUFFIXES

    # Build a miniature tree and prove a packaging change moves the source digest.
    (tmp_path / "packaging" / "windows").mkdir(parents=True)
    build_script = tmp_path / "packaging" / "windows" / "build_exe_scope.ps1"
    build_script.write_text("v1", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    before, _ = mod.source_tree_digest()
    build_script.write_text("v2", encoding="utf-8")
    after, _ = mod.source_tree_digest()
    assert before != after

    # Writing an attestation copies the gate report; verification detects a missing or altered copy.
    report = tmp_path / "gate.json"
    report.write_text(json.dumps({"release_decision": "RELEASE-CANDIDATE", "checks": {"a": {"result": "PASS"}}}), encoding="utf-8")
    out = tmp_path / "release_evidence" / "ATTESTATION.json"
    copy = tmp_path / "release_evidence" / "GATE_REPORT.json"
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(mod, "GATE_REPORT_COPY", copy)
    assert mod.main(["--gate-report", str(report)]) == 0
    assert copy.read_bytes() == report.read_bytes()
    assert mod.main(["--verify"]) == 0
    copy.write_text("tampered", encoding="utf-8")
    assert mod.main(["--verify"]) == 1
    copy.unlink()
    assert mod.main(["--verify"]) == 1


def test_reproducible_zip_verifier_enforces_hashes_for_lockfile() -> None:
    text = (ROOT / "scripts/reproducible_zip_verifier.py").read_text(encoding="utf-8")
    assert '"--require-hashes"' in text


# --- desktop toolchain pinning, wheelhouse build, signing ---

def test_desktop_requirements_are_exact_and_hash_locked() -> None:
    import re

    lines = [l for l in (ROOT / "packaging/windows/requirements-desktop.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    names = {l.split("==")[0] for l in lines}
    assert {"pywebview", "pyinstaller", "pefile", "pywin32-ctypes", "pythonnet"} <= names
    for l in lines:
        assert re.match(r"^[a-z0-9-]+==[0-9][^\s]* (--hash=sha256:[0-9a-f]{64} ?)+$", l), l[:80]
    assert ">=" not in "\n".join(lines)


def test_verify_desktop_pins_detects_mismatch(tmp_path) -> None:
    import importlib.util
    from importlib import metadata

    spec = importlib.util.spec_from_file_location("verify_desktop_pins", ROOT / "scripts/verify_desktop_pins.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    have = metadata.version("pytest")
    ok = tmp_path / "ok.txt"
    ok.write_text(f"pytest=={have} --hash=sha256:{'0' * 64}\n", encoding="utf-8")
    assert mod.main([str(ok)]) == 0
    bad = tmp_path / "bad.txt"
    bad.write_text("pytest==0.0.1\nnot-a-real-package-xyz==1.0\n", encoding="utf-8")
    assert mod.main([str(bad)]) == 1
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n", encoding="utf-8")
    assert mod.main([str(empty)]) == 1


def test_build_script_checks_pins_builds_wheelhouse_and_certifies() -> None:
    text = (ROOT / "packaging/windows/build_exe_scope.ps1").read_text(encoding="utf-8")
    assert "verify_desktop_pins.py" in text
    assert "pip download --require-hashes --only-binary=:all:" in text
    assert "offline_desktop_certification.py" in text


def test_signing_scripts_fail_closed() -> None:
    sign = (ROOT / "packaging/windows/sign_release.ps1").read_text(encoding="utf-8")
    verify = (ROOT / "packaging/windows/verify_release_signature.ps1").read_text(encoding="utf-8")
    assert "refusing to continue without a code-signing certificate" in sign
    assert "/fd" in sign and "/tr" in sign and "SHA256" in sign
    assert "verify_release_signature.ps1" in sign
    assert "-ne 'Valid'" in verify and "TimeStamperCertificate" in verify and "exit 1" in verify


def test_frontend_lock_next15_closure_is_consistent() -> None:
    import json

    pkgs = json.loads((ROOT / "apps/web/package-lock.json").read_text(encoding="utf-8"))["packages"]
    nxt = pkgs["node_modules/next"]["version"]
    assert nxt.startswith("15.")
    swc = {k: v["version"] for k, v in pkgs.items() if k.startswith("node_modules/@next/swc-")}
    assert swc and set(swc.values()) == {nxt}, swc
    assert pkgs["node_modules/@next/env"]["version"] == nxt
