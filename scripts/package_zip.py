"""
package_zip.py: Packages clean source-only AA-OS repository into destination zip files.
Excludes node_modules, .next, venvs, cache, runtime DBs, and non-source artifacts.
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
    ".git",
    ".idea",
    ".vscode",
    "temp_p2_test",
    "data_store",
}

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

def is_excluded_path(rel_path: str) -> bool:
    parts = rel_path.replace("/", "\\").split("\\")
    for part in parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
        if part.startswith(".next") or part.startswith(".venv") or part == ".pytest_cache":
            return True
    
    file_name = parts[-1]
    if file_name in EXCLUDE_EXACT_FILES:
        return True
    if file_name.lower().endswith(EXCLUDE_NAME_SUFFIXES):
        return True
    if file_name.lower().endswith(EXCLUDE_DB_SIDECAR_SUFFIXES):
        return True

    ext = os.path.splitext(file_name)[1].lower()
    if ext in EXCLUDE_EXTENSIONS:
        return True
        
    return False

def make_archive(dest_zip_path: str):
    print(f"Creating clean source archive: {dest_zip_path}...")
    temp_zip = dest_zip_path + ".tmp"
    total_files = 0
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(WORKSPACE_DIR):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and not d.startswith(".")]
            
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, WORKSPACE_DIR)
                
                if is_excluded_path(rel_path):
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
    destinations = [
        os.path.join(DOWNLOADS_DIR, "Minded_AAOS_v32_2026-09-26_session17_question_resolution_release_gate_fix.zip"),
        os.path.join(DOWNLOADS_DIR, "Minded_AAOS_v29_offline_desktop_certified_final.zip"),
        os.path.join(DOWNLOADS_DIR, "DataBase-AI-Platform.zip"),
        os.path.join(DOWNLOADS_DIR, "DataBase.zip"),
        os.path.join(WORKSPACE_DIR, "DataBase-AI-Platform.zip"),
        os.path.join(WORKSPACE_DIR, "DataBase.zip"),
    ]
    for dest in destinations:
        make_archive(dest)
    print("\nAll clean source archives successfully generated!")

if __name__ == "__main__":
    main()
