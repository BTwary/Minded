"""Build Native Windows Installer Setup (MindedSetup.exe) for Minded AA-OS.

Compiles native C# launcher, uninstaller, and setup wizard using Windows csc.exe,
embeds the complete offline application payload, and outputs a standalone MindedSetup.exe.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_DIR = Path(__file__).resolve().parent
CSC_COMPILER = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
DOWNLOADS_DIR = Path(os.path.expanduser("~/Downloads"))

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    ".next",
    "dist",
    "build",
    "data_store",
    "tests",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pkl",
    ".zip",
    ".log",
    ".tmp",
}


def compile_csharp(source_file: Path, output_file: Path, references: list[str] | None = None,
                   resources: list[tuple[Path, str]] | None = None) -> None:
    if not CSC_COMPILER.is_file():
        raise RuntimeError(f"C# compiler not found at {CSC_COMPILER}")

    cmd = [
        str(CSC_COMPILER),
        "/nologo",
        "/target:winexe",
        f"/out:{output_file}",
    ]

    for ref in references or []:
        cmd.append(f"/r:{ref}")

    for res_path, res_name in resources or []:
        cmd.append(f"/resource:{res_path},{res_name}")

    cmd.append(str(source_file))

    print(f"Compiling {source_file.name} -> {output_file.name}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Compilation of {source_file.name} failed:\n{res.stderr or res.stdout}")
    print(f" -> Successfully compiled {output_file.name}")


def create_payload_zip(target_zip: Path) -> int:
    print(f"Packaging clean application payload into {target_zip.name}...")
    file_count = 0
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            rel_dir = os.path.relpath(root, REPO_ROOT)
            if rel_dir.startswith("packaging\\windows\\installer"):
                continue

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in EXCLUDE_EXTS or f.endswith(".zip") or f.endswith(".exe"):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, REPO_ROOT)
                zf.write(full_path, rel_path)
                file_count += 1

    size_mb = target_zip.stat().st_size / (1024 * 1024)
    print(f" -> Payload packaged: {file_count} files ({size_mb:.2f} MB)")
    return file_count


def build_installer() -> Path:
    print("=" * 70)
    print("BUILDING MINDED ENTERPRISE DESKTOP INSTALLER (MindedSetup.exe)")
    print("=" * 70)

    launcher_cs = INSTALLER_DIR / "MindedLauncher.cs"
    launcher_exe = INSTALLER_DIR / "Minded.exe"
    uninstaller_cs = INSTALLER_DIR / "MindedUninstall.cs"
    uninstaller_exe = INSTALLER_DIR / "Uninstall.exe"
    setup_cs = INSTALLER_DIR / "MindedSetup.cs"
    setup_exe = INSTALLER_DIR / "MindedSetup.exe"
    payload_zip = INSTALLER_DIR / "payload.zip"

    # Step 1: Compile Launcher
    compile_csharp(launcher_cs, launcher_exe)

    # Step 2: Compile Uninstaller
    compile_csharp(uninstaller_cs, uninstaller_exe)

    # Step 3: Create payload ZIP
    create_payload_zip(payload_zip)

    # Step 4: Compile Setup Wizard with embedded resources
    compile_csharp(
        setup_cs,
        setup_exe,
        references=[
            "System.IO.Compression.dll",
            "System.IO.Compression.FileSystem.dll",
            "Microsoft.CSharp.dll",
        ],
        resources=[
            (payload_zip, "payload.zip"),
            (launcher_exe, "Minded.exe"),
            (uninstaller_exe, "Uninstall.exe"),
        ],
    )

    setup_size_mb = setup_exe.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Standalone installer generated: {setup_exe.name} ({setup_size_mb:.2f} MB)")

    # Step 5: Distribute to destination directories
    destinations = [
        DOWNLOADS_DIR / "MindedSetup.exe",
        REPO_ROOT / "packaging" / "windows" / "MindedSetup.exe",
        REPO_ROOT / "MindedSetup.exe",
    ]

    for dest in destinations:
        shutil.copy2(setup_exe, dest)
        print(f" -> Copied installer to: {dest}")

    # Clean up intermediate build artifacts
    if payload_zip.is_file():
        payload_zip.unlink()

    print("\n[SUCCESS] INSTALLATION SETUP BUILD COMPLETE!")
    return setup_exe


if __name__ == "__main__":
    build_installer()
