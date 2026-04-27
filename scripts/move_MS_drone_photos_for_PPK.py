"""
Emlid Studio's Drone Data Processing workflow expects a single image folder
containing only RGB photographs (.jpg files).
The .tif multispectral images in those folders interfere with that workflow.

This script crawls through the directory tree under ROOT_DIR, finds all .tif
files, and moves each one into an 'MS' subfolder within its current directory.
The result is that RGB .jpg images remain in place for Emlid Studio, while
multispectral .tif images are organized separately.

Update the ROOT_DIR variable to point to the top-level directory containing your photos


"""
from pathlib import Path
import shutil
import sys


ROOT_DIR = r"D:\SfM\20260326_DCC_photos"
MODE = "move"   # change to "copy" if you want to keep originals


def relocate_tifs(root_dir: str, mode: str = "move") -> None:
    """
    Crawl through root_dir and all subdirectories.
    For each .tif file found, place it into an 'MS' subfolder within its current parent directory.
    """
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    tif_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".tif"]

    for tif in tif_files:
        # Skip files already inside an MS folder
        if tif.parent.name == "MS":
            continue

        ms_dir = tif.parent / "MS"
        ms_dir.mkdir(exist_ok=True)

        destination = ms_dir / tif.name

        # Handle name collisions
        if destination.exists():
            stem = tif.stem
            suffix = tif.suffix
            i = 1
            while True:
                candidate = ms_dir / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    destination = candidate
                    break
                i += 1

        if mode == "move":
            shutil.move(str(tif), str(destination))
            print(f"Moved: {tif} -> {destination}")
        elif mode == "copy":
            shutil.copy2(str(tif), str(destination))
            print(f"Copied: {tif} -> {destination}")
        else:
            raise ValueError("mode must be 'move' or 'copy'")


if __name__ == "__main__":
    try:
        relocate_tifs(ROOT_DIR, MODE)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)