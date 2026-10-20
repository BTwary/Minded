"""
package_zip.py: Packages clean application source into a destination ZIP.

Two profiles (they are different artifacts and must not be confused):

  certification (default)  Self-certifying release bundle. Includes tests/, data/certification/
                           and the stress-result evidence, because tests/independent_release/
                           manifest.toml (the release gate) requires them. A ZIP in this profile
                           can run scripts/release_gate.py from a fresh extraction.
  runtime                  Slim runtime/source distribution. Excludes test suites, benchmark
                           datasets/results and release evidence. It CANNOT run the release gate.

Both profiles exclude generated frontend metadata, dependency installs, build outputs, caches,
runtime databases, secrets and binaries.

Usage: python scripts/package_zip.py [dest.zip] [--profile certification|runtime]
"""
import os
import zipfile

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")

EXCLUDE_DIR_NAMES = {
    "node_modules",
    ".next",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "coverage",
    "htmlcov",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".tox",
    ".git",
    ".idea",
    ".vscode",
    "temp_p2_test",
    "data_store",
    "bundle",
    "out",
}

# Excluded only from the slim ``runtime`` profile; the ``certification`` profile ships them.
RUNTIME_ONLY_EXCLUDE_DIR_NAMES = {"tests"}
RUNTIME_ONLY_EXCLUDE_RELATIVE_DIRS = {"data/certification", "release_evidence"}
RUNTIME_ONLY_EXCLUDE_EXACT_FILES = {"real_stress_results_clean.json", "real_stress_results_raw.json"}

PROFILES = ("certification", "runtime")

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".zip",
    ".tar",
    ".gz",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".tmp",
    ".log",
    ".pkl",
    ".exe",
    ".tsbuildinfo",
}

EXCLUDE_EXACT_FILES = {
    "autonomous_analyst.db",
    "test.db",
    ".env",
    "phase_release_report.json",
}
EXCLUDE_NAME_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
# SQLite write-ahead-log / shared-memory sidecar files (e.g.
# "autonomous_analyst.db-wal", "autonomous_analyst.db-shm") contain live,
# uncommitted database pages -- excluding the base ".db" file by exact name
# is not sufficient, since these sidecars are never matched by
# EXCLUDE_EXACT_FILES or EXCLUDE_EXTENSIONS and were leaking real data into
# otherwise "clean source" release archives.
EXCLUDE_DB_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

def _exclude_dir_names(profile: str) -> set:
    return EXCLUDE_DIR_NAMES | (RUNTIME_ONLY_EXCLUDE_DIR_NAMES if profile == "runtime" else set())


def _exclude_relative_dirs(profile: str) -> set:
    return set(RUNTIME_ONLY_EXCLUDE_RELATIVE_DIRS) if profile == "runtime" else set()


def is_excluded_path(rel_path: str, profile: str = "certification") -> bool:
    norm = rel_path.replace("\\", "/")
    if any(norm == d or norm.startswith(d + "/") for d in _exclude_relative_dirs(profile)):
        return True
    parts = norm.split("/")
    for part in parts:
        if part in _exclude_dir_names(profile):
            return True
        if part.startswith(".next") or part.startswith(".venv") or part == ".pytest_cache":
            return True
    
    file_name = parts[-1]
    if file_name in EXCLUDE_EXACT_FILES:
        return True
    if profile == "runtime" and file_name in RUNTIME_ONLY_EXCLUDE_EXACT_FILES:
        return True
    if file_name.lower().endswith(EXCLUDE_NAME_SUFFIXES):
        return True
    if file_name.lower().endswith(EXCLUDE_DB_SIDECAR_SUFFIXES):
        return True

    ext = os.path.splitext(file_name)[1].lower()
    if ext in EXCLUDE_EXTENSIONS:
        return True
        
    return False

def make_archive(dest_zip_path: str, profile: str = "certification"):
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    print(f"Creating clean source archive ({profile} profile): {dest_zip_path}...")
    temp_zip = dest_zip_path + ".tmp"
    total_files = 0
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(WORKSPACE_DIR):
            # Prune excluded directories
            parent_rel = os.path.relpath(root, WORKSPACE_DIR).replace(os.sep, "/")
            dirs[:] = [
                d for d in dirs
                if d not in _exclude_dir_names(profile)
                and not d.startswith(".")
                and os.path.normpath(os.path.join(parent_rel, d)).replace(os.sep, "/") not in _exclude_relative_dirs(profile)
            ]
            
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, WORKSPACE_DIR)
                
                if is_excluded_path(rel_path, profile):
                    continue
                
                zf.write(full_path, rel_path)
                total_files += 1

    if os.path.exists(dest_zip_path):
        os.remove(dest_zip_path)
    os.rename(temp_zip, dest_zip_path)
    file_size_kb = os.path.getsize(dest_zip_path) / 1024
    if file_size_kb > 1024:
        print(f" -> Completed {dest_zip_path}: {total_files} files packaged ({file_size_kb / 1024:.2f} MB).")
    else:
        print(f" -> Completed {dest_zip_path}: {total_files} files packaged ({file_size_kb:.1f} KB).")

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build a clean release source ZIP")
    ap.add_argument("dest_name", nargs="?", default="MindEd_AAOS_v1.0.0_complete_source_candidate.zip")
    ap.add_argument("--profile", choices=PROFILES, default="certification")
    args = ap.parse_args()
    dest = os.path.join(DOWNLOADS_DIR, args.dest_name) if not os.path.isabs(args.dest_name) else args.dest_name
    make_archive(dest, args.profile)
    print("\nClean source archive successfully generated!")

if __name__ == "__main__":
    main()
