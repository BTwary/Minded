"""Script to package the complete autonomous-data-analyst platform into a clean ZIP bundle."""
import os
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ZIP = os.path.join(PROJECT_ROOT, "..", "autonomous-data-analyst.zip")

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".next",
    ".git",
    ".idea",
    ".vscode",
    "dist",
    "build",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pkl",
    ".DS_Store",
}


def bundle_repository():
    print(f"Packaging project from: {PROJECT_ROOT}")
    print(f"Destination: {OUTPUT_ZIP}")

    total_files = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in EXCLUDE_EXTS or file.endswith(".zip"):
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, os.path.join(PROJECT_ROOT, ".."))
                zip_file.write(full_path, rel_path)
                total_files += 1

    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"\nSuccessfully created bundle with {total_files} files ({zip_size_mb:.2f} MB)!")
    print(f"Archive location: {OUTPUT_ZIP}")


if __name__ == "__main__":
    bundle_repository()
