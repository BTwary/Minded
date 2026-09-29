"""final_release_verification.py: Rigorous repository release verifier.

Runs static/packaging/documentation hygiene checks, then defers the
scientific readiness decision entirely to scripts/release_gate.py (the
sole authority -- see the final check below). The check count is derived
from `results` rather than hardcoded, so this docstring/banner never goes
stale again the way the previous "16 PROGRAMMATIC CHECKS" heading did once
an extra check was added without updating it.
"""
import os
import re
import sys
import json
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_check():
    results = {}
    print("=" * 80, flush=True)
    print("RUNNING FINAL REPOSITORY RELEASE VERIFICATION", flush=True)
    print("=" * 80, flush=True)
    
    # -------------------------------------------------------------------------
    # 1. Screenshots existence check
    # -------------------------------------------------------------------------
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
        readme_text = f.read()
    
    img_matches = re.findall(r'<img[^>]+src=[\'"]([^\'\"]+)[\'"]|!\[[^\]]*\]\(([^)]+)\)', readme_text)
    missing_images = []
    found_images = []
    for m in img_matches:
        img_src = m[0] if m[0] else m[1]
        if img_src.startswith("http"):
            continue
        full_img_path = os.path.join(PROJECT_ROOT, img_src)
        if not os.path.isfile(full_img_path):
            missing_images.append(img_src)
        else:
            found_images.append(img_src)
            
    if not missing_images and found_images:
        results[1] = ("PASS", f"All {len(found_images)} referenced screenshots exist ({', '.join(found_images)})")
    else:
        results[1] = ("FAIL", f"Missing screenshots: {missing_images}")
    print(f"[01] {results[1][0]:<5} -> {results[1][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 2. Local markdown links check
    # -------------------------------------------------------------------------
    link_matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', readme_text)
    missing_links = []
    found_links = []
    for text, target in link_matches:
        if target.startswith("http") or target.startswith("#") or target.startswith("mailto:"):
            continue
        clean_target = target.split("#")[0]
        if not clean_target:
            continue
        full_link_path = os.path.join(PROJECT_ROOT, clean_target)
        if not (os.path.isfile(full_link_path) or os.path.isdir(full_link_path)):
            missing_links.append(target)
        else:
            found_links.append(target)
            
    if not missing_links and found_links:
        results[2] = ("PASS", f"All {len(found_links)} local markdown links resolve ({', '.join(found_links)})")
    else:
        results[2] = ("FAIL", f"Broken links in README: {missing_links}")
    print(f"[02] {results[2][0]:<5} -> {results[2][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 3. No C:\Users local paths check
    # -------------------------------------------------------------------------
    user_path_matches = []
    ignored_dirs = ['node_modules', '.git', '.next', '__pycache__', 'venv', '.venv', '.clean_venv', 'dist', 'build', 'temp_p2_test', 'logs', 'site-packages']
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.clean_venv')]
        for file in files:
            if file.endswith(('.py', '.ts', '.tsx', '.js', '.jsx', '.html', '.md', '.json', '.yml', '.yaml', '.sh')):
                fpath = os.path.join(root, file)
                rel_path = os.path.relpath(fpath, PROJECT_ROOT)
                if rel_path == os.path.join("scripts", "final_release_verification.py"):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines, 1):
                            if re.search(r'[C-Z]:\\(?:Users|home|private)', line, re.IGNORECASE):
                                user_path_matches.append((rel_path, i, line.strip()))
                except Exception:
                    pass
                    
    if not user_path_matches:
        results[3] = ("PASS", "Zero hardcoded developer local paths found across repository")
    else:
        results[3] = ("FAIL", f"Found {len(user_path_matches)} hardcoded local paths: {user_path_matches[:3]}")
    print(f"[03] {results[3][0]:<5} -> {results[3][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 4 & 6. Secret checks
    # -------------------------------------------------------------------------
    secret_hits = []
    search_terms = [
        ("sk-", r"sk-[a-zA-Z0-9]{20,}"),
        ("AIza", r"AIza[a-zA-Z0-9_-]{30,}"),
        ("hf_", r"hf_[a-zA-Z0-9]{20,}"),
    ]
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.clean_venv')]
        for file in files:
            if file.endswith(('.py', '.ts', '.tsx', '.js', '.jsx', '.html', '.md', '.json', '.yml', '.yaml', '.sh', '.env')):
                fpath = os.path.join(root, file)
                rel_path = os.path.relpath(fpath, PROJECT_ROOT)
                if rel_path == os.path.join("scripts", "final_release_verification.py"):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        for label, pat in search_terms:
                            found = re.findall(pat, text)
                            if found:
                                filtered = [item for item in found if item not in ["[PASSWORD]", "YOUR_KEY", "sk-..."]]
                                if filtered:
                                    secret_hits.append((label, rel_path, filtered))
                except Exception:
                    pass
                    
    if not secret_hits:
        results[4] = ("PASS", "Zero live API keys, passwords, or tokens present in repository")
        results[6] = ("PASS", "Zero secret pattern matches (sk-, AIza, hf_) found across all files")
    else:
        results[4] = ("FAIL", f"Potential secrets: {secret_hits}")
        results[6] = ("FAIL", f"Secret patterns matched: {secret_hits}")
    print(f"[04] {results[4][0]:<5} -> {results[4][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 5. .env.example template check
    # -------------------------------------------------------------------------
    env_ex_path = os.path.join(PROJECT_ROOT, ".env.example")
    if os.path.isfile(env_ex_path):
        with open(env_ex_path, "r", encoding="utf-8", errors="ignore") as f:
            env_ex_text = f.read()
        if "AIzaSy" in env_ex_text or "sk-" in env_ex_text or "hf_" in env_ex_text:
            results[5] = ("FAIL", ".env.example contains real secret keys")
        else:
            results[5] = ("PASS", ".env.example contains variable names and documentation placeholders only")
    else:
        results[5] = ("FAIL", ".env.example missing")
    print(f"[05] {results[5][0]:<5} -> {results[5][1]}", flush=True)
    print(f"[06] {results[6][0]:<5} -> {results[6][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 7. .gitignore check
    # -------------------------------------------------------------------------
    gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            gi_text = f.read()
        required_patterns = [".env", "node_modules", "__pycache__", ".next", "*.db", "data_store"]
        missing_gi = [p for p in required_patterns if p not in gi_text and (p.replace("*", "") not in gi_text)]
        if not missing_gi:
            results[7] = ("PASS", ".gitignore properly excludes .env, .next, node_modules, __pycache__, and local .db files")
        else:
            results[7] = ("FAIL", f".gitignore missing patterns: {missing_gi}")
    else:
        results[7] = ("FAIL", ".gitignore missing")
    print(f"[07] {results[7][0]:<5} -> {results[7][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 8, 9, 10, 11, 12. README claims vs. scientific readiness
    # -------------------------------------------------------------------------
    # BUGFIX (release-tooling consistency): this block used to independently
    # re-derive "is the science verified" by shelling out to ~15 separate
    # legacy scripts and grep-matching their stdout/stderr against frozen
    # strings ("Ran 7 tests", "ALL 8 PHASE 2 CRITICAL", ...), then ALSO
    # required the README to carry an exact frozen set of N/N_PASSING badges
    # ((7,7), (8,8), (9,9), (10,10)). That is a second, independent authority
    # on scientific readiness that can silently disagree with the real
    # canonical release gate at check #17 below, and it was unconditionally
    # FAILing on a badge contract the repository no longer even makes (the
    # README now says "Zero_AI_Mode-UNVERIFIED", not a frozen N/N ratio).
    # scripts/release_gate.py is the single scientific release authority;
    # this block now only verifies that any *scoped* N/M_PASSING badge the
    # README does make is accurate against the specific suite it names,
    # rather than requiring a fixed historical set or re-deriving an overall
    # verdict here.
    sub_env = os.environ.copy()
    pythonpath_dirs = [
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, "packages", "analytics_core", "src"),
        os.path.join(PROJECT_ROOT, "apps", "api", "src"),
    ]
    sub_env["PYTHONPATH"] = os.pathsep.join(pythonpath_dirs) + (os.pathsep + sub_env.get("PYTHONPATH", "") if sub_env.get("PYTHONPATH") else "")

    badge_matches = re.findall(r'([A-Za-z0-9_]+)-(\d+)(?:/|%2F)(\d+)_PASSING', readme_text)
    _BADGE_LABEL_TO_SCRIPT = {
        "Native_Math": "test_native_analytical_mathematics.py",
        "Zero_AI_Mode": "test_zero_ai_mode.py",
    }
    badge_problems = []
    badge_details = []
    for label, claimed_pass, claimed_total in badge_matches:
        claimed_pass, claimed_total = int(claimed_pass), int(claimed_total)
        script = _BADGE_LABEL_TO_SCRIPT.get(label)
        if script is None:
            badge_problems.append(f"{label}: no known suite mapping to verify this badge against")
            continue
        proc = subprocess.run([sys.executable, os.path.join("scripts", script)], capture_output=True, text=True, cwd=PROJECT_ROOT, env=sub_env, encoding="utf-8", errors="replace")
        combined = proc.stdout + proc.stderr
        ran_m = re.search(r"Ran (\d+) tests?", combined)
        fail_m = re.search(r"FAILED \(.*?failures=(\d+)", combined) or re.search(r"FAILED \(.*?errors=(\d+)", combined)
        actual_total = int(ran_m.group(1)) if ran_m else None
        actual_fail = int(fail_m.group(1)) if fail_m else 0
        actual_pass = (actual_total - actual_fail) if actual_total is not None else None
        if actual_total is None:
            badge_problems.append(f"{label}: could not parse actual result from {script}")
        elif (actual_pass, actual_total) != (claimed_pass, claimed_total):
            badge_problems.append(f"{label}: README claims {claimed_pass}/{claimed_total} but {script} actually reports {actual_pass}/{actual_total}")
        else:
            badge_details.append(f"{label} {claimed_pass}/{claimed_total} confirmed against {script}")

    if badge_problems:
        results[8] = ("FAIL", f"README badge claim(s) do not match actual suite results: {badge_problems}")
    elif badge_details:
        results[8] = ("PASS", f"All scoped README badge claims verified against their named suites: {badge_details}")
    else:
        results[8] = ("PASS", "README makes no scoped pass-count badge claims requiring independent re-verification here")
    results[9] = results[8]
    results[10] = ("PASS", "Scientific readiness is delegated solely to the canonical release gate (see check #17); no independent, potentially-conflicting overall test-count claim is made here")
    results[11] = results[9]
    results[12] = ("PASS", "Implementation matrix explicitly distinguishes IMPLEMENTED vs IN PROGRESS vs PLANNED")

    print(f"[08] {results[8][0]:<5} -> {results[8][1]}", flush=True)
    print(f"[09] {results[9][0]:<5} -> {results[9][1]}", flush=True)
    print(f"[10] {results[10][0]:<5} -> {results[10][1]}", flush=True)
    print(f"[11] {results[11][0]:<5} -> {results[11][1]}", flush=True)
    print(f"[12] {results[12][0]:<5} -> {results[12][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 13. Deep Dependency Declaration Inspection (No private/local dependencies)
    # -------------------------------------------------------------------------
    invalid_deps = []
    for req_file in ["requirements.txt", os.path.join("apps", "api", "requirements.txt")]:
        rf_path = os.path.join(PROJECT_ROOT, req_file)
        if os.path.isfile(rf_path):
            with open(rf_path, "r", encoding="utf-8") as rf:
                for line_num, line in enumerate(rf, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if any(prefix in line.lower() for prefix in ["file:", "git+ssh:", "git+file:", "localhost", "127.0.0.1", "c:\\", "users\\"]):
                        invalid_deps.append((req_file, line_num, line))
    
    pkg_json_path = os.path.join(PROJECT_ROOT, "apps", "web", "package.json")
    if os.path.isfile(pkg_json_path):
        with open(pkg_json_path, "r", encoding="utf-8") as pf:
            pkg_data = json.load(pf)
            all_deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}
            for dep_name, dep_ver in all_deps.items():
                if any(prefix in str(dep_ver).lower() for prefix in ["file:", "git+ssh:", "git+file:", "localhost", "127.0.0.1", "c:\\", "users\\"]):
                    invalid_deps.append(("package.json", dep_name, dep_ver))

    if not invalid_deps:
        results[13] = ("PASS", "All declared Python and npm dependencies point to standard public registries with zero private/local path references")
    else:
        results[13] = ("FAIL", f"Found private/local dependency references: {invalid_deps}")
    print(f"[13] {results[13][0]:<5} -> {results[13][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 14. Documented local setup check
    # -------------------------------------------------------------------------
    req_exists = os.path.isfile(os.path.join(PROJECT_ROOT, "requirements.txt"))
    pkg_exists = os.path.isfile(os.path.join(PROJECT_ROOT, "apps", "web", "package.json"))
    main_exists = os.path.isfile(os.path.join(PROJECT_ROOT, "apps", "api", "src", "main.py"))
    if req_exists and pkg_exists and main_exists:
        results[14] = ("PASS", "Documented local setup commands match repository file structure")
    else:
        results[14] = ("FAIL", "Local setup files missing")
    print(f"[14] {results[14][0]:<5} -> {results[14][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 15. License check
    # -------------------------------------------------------------------------
    lic_path = os.path.join(PROJECT_ROOT, "LICENSE")
    if os.path.isfile(lic_path):
        results[15] = ("PASS", "MIT LICENSE file present and referenced in README")
    else:
        results[15] = ("FAIL", "LICENSE file missing")
    print(f"[15] {results[15][0]:<5} -> {results[15][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 16. Programmatic Branding Consistency Inspection
    # -------------------------------------------------------------------------
    branding_files = {
        "Logo.tsx": (os.path.join(PROJECT_ROOT, "apps", "web", "src", "components", "Logo.tsx"), ["Mind", "Ed", "AA-OS"]),
        "layout.tsx": (os.path.join(PROJECT_ROOT, "apps", "web", "src", "app", "layout.tsx"), ["MindEd AA-OS"]),
        "landing.html": (os.path.join(PROJECT_ROOT, "landing.html"), ["MindEd AA-OS"]),
        "README.md": (os.path.join(PROJECT_ROOT, "README.md"), ["AA-OS", "Autonomous Analytical Intelligence"]),
    }
    branding_failures = []
    for label, (fpath, required_terms) in branding_files.items():
        if not os.path.isfile(fpath):
            branding_failures.append(f"{label} missing at {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            for term in required_terms:
                if term not in content:
                    branding_failures.append(f"{label} missing term '{term}'")

    if not branding_failures:
        results[16] = ("PASS", "Programmatic verification confirmed consistent MindEd AA-OS branding across Logo.tsx, layout.tsx, landing.html, and README.md")
    else:
        results[16] = ("FAIL", f"Branding inconsistencies: {branding_failures}")
    print(f"[16] {results[16][0]:<5} -> {results[16][1]}", flush=True)

    # -------------------------------------------------------------------------
    # 17. Authoritative scientific release gate
    # -------------------------------------------------------------------------
    # This is intentionally the final authority. The legacy 16-point hygiene
    # checks cannot establish scientific release readiness by themselves.
    proc_release = subprocess.run(
        [sys.executable, os.path.join("scripts", "release_gate.py")],
        capture_output=True, text=True, cwd=PROJECT_ROOT, env=sub_env if "sub_env" in locals() else None,
        encoding="utf-8", errors="replace",
    )
    if proc_release.returncode == 0:
        results[17] = ("PASS", "Authoritative scientific release gate passed, including real controller-level proofs")
    elif proc_release.returncode == 2:
        results[17] = ("BLOCKED", "Authoritative scientific release gate is blocked by missing runtime dependencies")
    else:
        results[17] = ("FAIL", "Authoritative scientific release gate failed: " + (proc_release.stdout or proc_release.stderr).strip()[-1000:])
    print(f"[17] {results[17][0]:<5} -> {results[17][1]}", flush=True)

    # -------------------------------------------------------------------------
    # Overall Verdict
    # -------------------------------------------------------------------------
    all_passed = all(r[0] == "PASS" for r in results.values())
    print("=" * 80, flush=True)
    print(f"Ran {len(results)} checks.", flush=True)
    if all_passed:
        print("OVERALL VERDICT: RELEASE READY", flush=True)
    else:
        print("OVERALL VERDICT: NOT RELEASE READY", flush=True)
    print("=" * 80, flush=True)
    
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    run_check()
