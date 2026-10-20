"""Build Lightweight Web/Bootstrapper Installer (MindedSetup.exe) for MindEd AA-OS.

Outputs:
1. MindedSetup.exe (~400 KB): Lightweight online/bootstrapper installer.
   During installation, it downloads the release archive (or uses a local package),
   expands the components, verifies WebView2, creates desktop shortcuts, and launches the app.
2. MindedAAOS-Windows-x64.zip: The distributable release payload to attach to GitHub Releases.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_DIR = Path(__file__).resolve().parent
CSC_COMPILER = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
DOWNLOADS_DIR = Path(os.path.expanduser("~/Downloads"))
BUNDLE_DIR = REPO_ROOT / "packaging" / "windows" / "bundle" / "MindedAAOS"


def compile_csharp(source_file: Path, output_file: Path, references: list[str] | None = None,
                   resources: list[tuple[Path, str]] | None = None,
                   extra_sources: list[Path] | None = None) -> None:
    if not CSC_COMPILER.is_file():
        raise RuntimeError(f"C# compiler not found at {CSC_COMPILER}")

    cmd = [
        str(CSC_COMPILER),
        "/nologo",
        "/target:winexe",
        "/optimize+",
        f"/out:{output_file}",
    ]

    for ref in references or []:
        cmd.append(f"/r:{ref}")

    for res_path, res_name in resources or []:
        cmd.append(f"/resource:{res_path},{res_name}")

    cmd.append(str(source_file))
    cmd.extend(str(extra) for extra in extra_sources or [])

    print(f"Compiling {source_file.name} -> {output_file.name}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Compilation of {source_file.name} failed:\n{res.stderr or res.stdout}")
    print(f" -> Successfully compiled {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB)")


def package_component_zip(target_zip: Path) -> None:
    if not BUNDLE_DIR.is_dir():
        raise RuntimeError(f"Desktop bundle not found at {BUNDLE_DIR}. Build the bundle first.")

    target_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating release component archive: {target_zip.name}...")
    file_count = 0
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for path in sorted(BUNDLE_DIR.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(BUNDLE_DIR).as_posix()
            zf.write(path, arcname=relative)
            file_count += 1
    size_mb = target_zip.stat().st_size / (1024 * 1024)
    print(f" -> Component archive created: {file_count} files ({size_mb:.2f} MB)")


def main():
    print("=" * 70)
    print("BUILDING LIGHTWEIGHT WEB/BOOTSTRAPPER SETUP (MindedSetup.exe)")
    print("=" * 70)

    launcher_cs = INSTALLER_DIR / "MindedLauncher.cs"
    launcher_exe = INSTALLER_DIR / "Minded.exe"
    uninstaller_cs = INSTALLER_DIR / "MindedUninstall.cs"
    uninstaller_exe = INSTALLER_DIR / "Uninstall.exe"
    setup_cs = INSTALLER_DIR / "MindedSetup.cs"
    setup_exe = INSTALLER_DIR / "MindedSetup.exe"
    release_zip = INSTALLER_DIR / "MindedAAOS-Windows-x64.zip"

    # 1. Compile launcher
    compile_csharp(launcher_cs, launcher_exe)

    # 2. Compile uninstaller
    compile_csharp(uninstaller_cs, uninstaller_exe)

    # 3. Create the release archive for online hosting/download
    package_component_zip(release_zip)

    # 4. Compile the small web installer (~400 KB)
    compile_csharp(
        setup_cs,
        setup_exe,
        references=[
            "System.IO.Compression.dll",
            "System.IO.Compression.FileSystem.dll",
            "Microsoft.CSharp.dll",
        ],
        resources=[
            (launcher_exe, "Minded.exe"),
            (uninstaller_exe, "Uninstall.exe"),
        ],
    )

    size_kb = setup_exe.stat().st_size / 1024
    print(f"\n[OK] Small Bootstrapper Installer Built: {setup_exe.name} ({size_kb:.1f} KB)")

    # 5. Distribute installer and package to root and Downloads
    destinations_setup = [
        DOWNLOADS_DIR / "MindedSetup.exe",
        REPO_ROOT / "MindedSetup.exe",
        REPO_ROOT / "packaging" / "windows" / "MindedSetup.exe",
    ]
    for dest in destinations_setup:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(setup_exe, dest)
        print(f" -> Copied installer to: {dest}")

    destinations_zip = [
        DOWNLOADS_DIR / "MindedAAOS-Windows-x64.zip",
        REPO_ROOT / "packaging" / "windows" / "bundle" / "MindedAAOS-Windows-x64.zip",
    ]
    for dest in destinations_zip:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(release_zip, dest)
        print(f" -> Copied component archive to: {dest}")

    print("\n[SUCCESS] LIGHTWEIGHT INSTALLER CREATION COMPLETE!")


if __name__ == "__main__":
    main()
