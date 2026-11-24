"""
Compress an RGB orthomosaic to approximately a target file size.

Strategy
--------
1. Preserve raster dimensions, pixel size, extent, CRS, and georeferencing.
2. Export tiled GeoTIFFs using JPEG compression.
3. Iteratively adjust JPEG quality to approach the requested target size.
4. Keep the highest JPEG quality that satisfies the target size.

Notes
-----
- Intended primarily for 8-bit RGB orthomosaics.
- JPEG compression is lossy.
- The resulting file size will not match the target exactly.
- For quantitative multispectral imagery or DEMs, use lossless compression instead.

Requirements
------------
pip install rasterio
"""

from pathlib import Path
import os
import shutil
import tempfile

import rasterio


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

INPUT_TIF = Path(r"C:\L\Lichen\Lichen - Documents\Projects\20240004_MAMP Development (Cascadia CD)\07_GIS\Data_out\SfM Products\20260713\RC Ortho 20260713 EPSG 6596.tif")
OUTPUT_TIF = Path(r"C:\L\Lichen\Lichen - Documents\Projects\20240004_MAMP Development (Cascadia CD)\07_GIS\Data_out\SfM Products\20260713\RC Ortho 20260713 EPSG 6596_reduced.tif")

TARGET_SIZE_MB = 500

# JPEG quality search range.
# 100 = highest quality / largest file.
# Lower values increase compression.
MIN_JPEG_QUALITY = 75
MAX_JPEG_QUALITY = 95

# Accept output within this percentage below the target.
TARGET_TOLERANCE = 0.03

# GeoTIFF tile size.
BLOCK_SIZE = 512


# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------

def bytes_to_mb(n_bytes):
    """Convert bytes to decimal MB."""
    return n_bytes / 1_000_000


def get_file_size_mb(path):
    """Return file size in MB."""
    return bytes_to_mb(path.stat().st_size)


def write_compressed_tif(src_path, dst_path, jpeg_quality):
    """
    Write a JPEG-compressed tiled GeoTIFF while preserving raster geometry.
    """

    with rasterio.open(src_path) as src:

        if src.dtypes[0] != "uint8":
            raise ValueError(
                f"Input datatype is {src.dtypes[0]}. "
                "JPEG-compressed GeoTIFF is primarily appropriate for uint8 "
                "RGB imagery."
            )

        if src.count < 3:
            raise ValueError(
                f"Input contains only {src.count} band(s). "
                "This script is intended for RGB orthomosaics."
            )

        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            compress="JPEG",
            jpeg_quality=jpeg_quality,
            tiled=True,
            blockxsize=BLOCK_SIZE,
            blockysize=BLOCK_SIZE,
            interleave="pixel",
            BIGTIFF="IF_SAFER",
        )

        # Remove settings that can conflict with JPEG output.
        profile.pop("predictor", None)

        with rasterio.open(dst_path, "w", **profile) as dst:

            # Process by raster blocks instead of loading the entire
            # orthomosaic into memory.
            for _, window in src.block_windows(1):
                data = src.read(window=window)
                dst.write(data, window=window)

            # Preserve dataset-level tags.
            dst.update_tags(**src.tags())

            # Preserve band-level tags where available.
            for band in range(1, src.count + 1):
                dst.update_tags(band, **src.tags(band))


def compress_to_target(
    input_tif,
    output_tif,
    target_size_mb,
    min_quality=20,
    max_quality=95,
    tolerance=0.03,
):
    """
    Find the highest JPEG quality that produces a file no larger than
    approximately the requested target size.
    """

    input_tif = Path(input_tif)
    output_tif = Path(output_tif)

    if not input_tif.exists():
        raise FileNotFoundError(input_tif)

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    original_size = get_file_size_mb(input_tif)

    print(f"Input:       {input_tif}")
    print(f"Input size:  {original_size:,.1f} MB")
    print(f"Target size: {target_size_mb:,.1f} MB")
    print()

    target_lower = target_size_mb * (1 - tolerance)

    best_path = None
    best_quality = None
    best_size = None

    with tempfile.TemporaryDirectory(
        dir=output_tif.parent
    ) as temp_dir:

        temp_dir = Path(temp_dir)

        low = min_quality
        high = max_quality

        while low <= high:

            quality = (low + high) // 2

            temp_tif = temp_dir / f"q{quality}.tif"

            print(f"Testing JPEG quality {quality}...", end=" ")

            write_compressed_tif(
                input_tif,
                temp_tif,
                jpeg_quality=quality,
            )

            size_mb = get_file_size_mb(temp_tif)

            print(f"{size_mb:,.1f} MB")

            if size_mb <= target_size_mb:

                # This quality works. Save it as the current best.
                if (
                    best_quality is None
                    or quality > best_quality
                ):
                    best_quality = quality
                    best_size = size_mb
                    best_path = temp_tif

                # Try higher image quality.
                low = quality + 1

            else:
                # File is too large. Increase compression.
                high = quality - 1

        if best_path is None:
            # Even minimum quality is too large.
            min_test = temp_dir / "minimum_quality.tif"

            print()
            print(
                f"Testing minimum JPEG quality "
                f"{min_quality}..."
            )

            write_compressed_tif(
                input_tif,
                min_test,
                jpeg_quality=min_quality,
            )

            minimum_size = get_file_size_mb(min_test)

            raise RuntimeError(
                "\nTarget cannot be reached without changing "
                "the raster resolution.\n"
                f"Size at JPEG quality {min_quality}: "
                f"{minimum_size:,.1f} MB\n"
                f"Requested target: {target_size_mb:,.1f} MB"
            )

        # The TemporaryDirectory will be deleted, so copy the best result.
        shutil.copy2(best_path, output_tif)

    final_size = get_file_size_mb(output_tif)

    print()
    print("----------------------------------------")
    print("Finished")
    print("----------------------------------------")
    print(f"JPEG quality: {best_quality}")
    print(f"Output size:  {final_size:,.1f} MB")
    print(f"Target size:  {target_size_mb:,.1f} MB")
    print(
        f"Reduction:    "
        f"{100 * (1 - final_size / original_size):.1f}%"
    )
    print(f"Output:       {output_tif}")

    if final_size < target_lower:
        print()
        print(
            "Note: the closest available JPEG quality produced a file "
            "somewhat smaller than the requested target."
        )


# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    compress_to_target(
        INPUT_TIF,
        OUTPUT_TIF,
        TARGET_SIZE_MB,
        min_quality=MIN_JPEG_QUALITY,
        max_quality=MAX_JPEG_QUALITY,
        tolerance=TARGET_TOLERANCE,
    )