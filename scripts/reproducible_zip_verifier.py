"""reproducible_zip_verifier.py: Extracts ZIP, creates a fresh venv, installs dependencies, and runs all suites."""
import os
import sys
import time
import shutil
import zipfile
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Accept the release ZIP as a CLI argument (`python reproducible_zip_verifier.py
# path/to/release.zip`) so this tool isn't pinned to one developer's
# ~/Downloads directory and a historical artifact filename. Falls back to the
# old default only when no argument is given, for backward compatibility.
_DEFAULT_ZIP_PATH = os.path.expanduser(r"~/Downloads/DataBase-AI-Platform.zip")
ZIP_PATH = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_ZIP_PATH
TIMESTAMP = int(time.time())
EXTRACT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", f"reproducible_fresh_venv_{TIMESTAMP}"))

def main():
    if not os.path.isfile(ZIP_PATH):
        print(f"ERROR: release ZIP not found at: {ZIP_PATH}", flush=True)
        print(f"Usage: python {os.path.basename(__file__)} <path-to-release.zip>", flush=True)
        sys.exit(1)
    print("=" * 80, flush=True)
    print("REPRODUCIBLE ISOLATED VIRTUAL ENVIRONMENT RELEASE VERIFICATION", flush=True)
    print(f"ZIP Source: {ZIP_PATH}", flush=True)
    print(f"Target Clean Workspace: {EXTRACT_DIR}", flush=True)
    print("=" * 80, flush=True)

    # 1. Extract ZIP
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
    print(f"==> Step 1: Extracted {len(os.listdir(EXTRACT_DIR))} top-level entries into clean directory.", flush=True)

    # 2. Create fresh virtual environment inside EXTRACT_DIR
    venv_dir = os.path.join(EXTRACT_DIR, ".clean_venv")
    print(f"==> Step 2: Creating fresh isolated Python virtual environment at: {venv_dir}", flush=True)
    venv_create_proc = subprocess.run([sys.executable, "-m", "venv", venv_dir], capture_output=True, text=True)
    if venv_create_proc.returncode != 0:
        print(f"FAILED to create venv: {venv_create_proc.stderr}", flush=True)
        sys.exit(1)

    # Determine venv Python executable
    if sys.platform == "win32":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    if not os.path.isfile(venv_python):
        print(f"FAILED: Virtual environment Python not found at {venv_python}", flush=True)
        sys.exit(1)

    # 3. Query Python version and executable info from isolated venv
    ver_proc = subprocess.run([venv_python, "--version"], capture_output=True, text=True)
    python_version = ver_proc.stdout.strip() or ver_proc.stderr.strip()
    print(f"==> Step 3: Verified Isolated Python: {venv_python} ({python_version})", flush=True)

    # 4. Install declared requirements in isolated venv using authoritative lockfile if present
    lock_path = os.path.join(EXTRACT_DIR, "requirements-lock.txt")
    req_path = lock_path if os.path.isfile(lock_path) else os.path.join(EXTRACT_DIR, "requirements.txt")
    print(f"==> Step 4: Installing dependencies from authoritative source {req_path} into isolated venv...", flush=True)
    pip_install_proc = subprocess.run([venv_python, "-m", "pip", "install", "-r", req_path], text=True)
    if pip_install_proc.returncode != 0:
        print(f"FAILED: Dependency installation failed with exit code {pip_install_proc.returncode}", flush=True)
        sys.exit(1)
    else:
        print(f"==> Step 4 Completed: Dependencies installed successfully into clean venv.", flush=True)

    # 5. Check additional dependencies in apps/api/requirements.txt if present and lockfile was not used
    api_req = os.path.join(EXTRACT_DIR, "apps", "api", "requirements.txt")
    if os.path.isfile(api_req) and not os.path.isfile(lock_path):
        print(f"==> Step 5: Checking/installing {api_req}...", flush=True)
        pip_api_proc = subprocess.run([venv_python, "-m", "pip", "install", "-r", api_req], text=True)
        if pip_api_proc.returncode != 0:
            print(f"FAILED: API dependency installation failed with exit code {pip_api_proc.returncode}", flush=True)
            sys.exit(1)

    # 5b. Verify environment consistency with pip check
    print("==> Step 5b: Running pip check to verify dependency consistency...", flush=True)
    pip_check_proc = subprocess.run([venv_python, "-m", "pip", "check"], capture_output=True, text=True)
    if pip_check_proc.returncode != 0:
        print(f"pip check warning / issue: {pip_check_proc.stdout.strip()} {pip_check_proc.stderr.strip()}", flush=True)
    else:
        print("==> Step 5b: pip check passed (clean dependency graph).", flush=True)

    # Setup isolated PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        EXTRACT_DIR,
        os.path.join(EXTRACT_DIR, "packages", "analytics_core", "src"),
        os.path.join(EXTRACT_DIR, "apps", "api", "src"),
    ])
    env["VIRTUAL_ENV"] = venv_dir

    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    steps = [
        ("Suite 1: Mode 1 Free / Zero AI API (7 Tests)", [venv_python, os.path.join("scripts", "test_zero_ai_mode.py")], EXTRACT_DIR),
        ("Suite 2: AI Fallback & Resilience (5 Tests)", [venv_python, os.path.join("scripts", "test_ai_fallback.py")], EXTRACT_DIR),
        ("Suite 3: Native Analytical Mathematics (10 Tests)", [venv_python, os.path.join("scripts", "test_native_analytical_mathematics.py")], EXTRACT_DIR),
        ("Suite 4: Data Quality & Semantic Gate (7 Tests)", [venv_python, os.path.join("scripts", "test_data_quality_and_semantic_gate.py")], EXTRACT_DIR),
        ("Suite 5: Golden Deterministic Investigation (1 Test)", [venv_python, os.path.join("scripts", "test_golden_deterministic_investigation.py")], EXTRACT_DIR),
        ("Suite 6: Unified Scientific Loop (7 Tests)", [venv_python, os.path.join("scripts", "test_unified_scientific_loop.py")], EXTRACT_DIR),
        ("Suite 7: Phase 2 Kernel Acceptance (8 Tests)", [venv_python, os.path.join("scripts", "test_phase2_autonomous_kernel.py")], EXTRACT_DIR),
        ("Suite 8: Golden Adaptive Investigation (9 Tests)", [venv_python, os.path.join("scripts", "test_golden_adaptive_investigation.py")], EXTRACT_DIR),
        ("Suite 9: Unified Canonical Intelligence Loop (1 Test)", [venv_python, os.path.join("scripts", "test_unified_canonical_loop.py")], EXTRACT_DIR),
        ("Suite 10: Generalization & Multi-Domain Benchmark (6 Tests)", [venv_python, os.path.join("scripts", "test_aaos_generalization.py")], EXTRACT_DIR),
        ("Suite 11: Canonical Controller E2E Generalization Benchmark (8 Tests)", [venv_python, os.path.join("scripts", "test_aaos_generalization_e2e.py")], EXTRACT_DIR),
        ("Suite 12: Phase 7 Enterprise Guardrails & Remediation (4 Tests)", [venv_python, os.path.join("scripts", "test_phase7_enterprise_guardrails.py")], EXTRACT_DIR),
        ("Suite 13: Phase 8 Business Context & Prescriptive Action (4 Tests)", [venv_python, os.path.join("scripts", "test_phase8_business_context.py")], EXTRACT_DIR),
        ("Suite 14: Phase 8 Proactive Drift & State Forking (8 Tests)", [venv_python, os.path.join("scripts", "test_phase8_proactive_forking.py")], EXTRACT_DIR),
        ("Suite 15: Phase 9 Causal Discovery & MIAP Protocol (5 Tests)", [venv_python, os.path.join("scripts", "test_phase9_causal_discovery.py")], EXTRACT_DIR),
        ("Compileall Bytecode Validation", [venv_python, "-m", "compileall", "-q", "."], EXTRACT_DIR),
        ("Web Dependencies (npm ci)", [npm_cmd, "ci", "--prefer-offline"], os.path.join(EXTRACT_DIR, "apps", "web")),
        ("Next.js Web Production Build", [npm_cmd, "run", "build"], os.path.join(EXTRACT_DIR, "apps", "web")),
        ("Final 16-Point Programmatic Verifier", [venv_python, os.path.join("scripts", "final_release_verification.py")], EXTRACT_DIR),
    ]

    summary = []
    for name, cmd, cwd in steps:
        print("\n" + "=" * 80, flush=True)
        print(f"RUNNING: {name}", flush=True)
        print(f"COMMAND: {' '.join(cmd)} (cwd={cwd})", flush=True)
        print("=" * 80, flush=True)
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        if p.stdout:
            print(p.stdout.strip(), flush=True)
        if p.stderr:
            print(p.stderr.strip(), flush=True)
        print(f"EXIT CODE: {p.returncode}", flush=True)
        status = "PASS" if p.returncode == 0 else "FAIL"
        summary.append((name, status, p.returncode))

    print("\n" + "=" * 80, flush=True)
    print("FRESH ISOLATED VIRTUAL ENVIRONMENT AUDIT SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Python Executable : {venv_python}", flush=True)
    print(f"Python Version    : {python_version}", flush=True)
    print(f"Dependency Source : {req_path}", flush=True)
    print("-" * 80, flush=True)
    all_ok = True
    for name, status, code in summary:
        print(f"[{status:<4}] (Exit Code {code}) -> {name}", flush=True)
        if status != "PASS":
            all_ok = False

    print("=" * 80, flush=True)
    if all_ok:
        print("FINAL VERDICT: RELEASE READY", flush=True)
    else:
        print("FINAL VERDICT: NOT RELEASE READY", flush=True)
    print("=" * 80, flush=True)

    # Cleanup test workspace
    try:
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
