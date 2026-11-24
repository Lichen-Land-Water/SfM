"""Transfer Emlid Studio RGB position corrections to DJI Mavic 3M TIFFs.

Run this file directly in VS Code. Edit USER SETTINGS below if necessary.
Requires Python 3.10+ and ExifTool (https://exiftool.org/); no Python packages.
On Windows, extract the ExifTool Windows distribution, keep its companion
files, rename exiftool(-k).exe to exiftool.exe, and put it beside this script
or set EXIFTOOL_PATH below. A copy already on PATH also works.

METHOD, per exposure and per band, in Earth-centered Cartesian coordinates:
    corrected_MS = corrected_RGB + (original_MS - original_RGB)

This retains the original DJI sensor-center offset. Emlid's RTK-drone mode
already applies the MRK antenna-to-RGB offset; do not apply that offset again.
The method assumes the original RGB and MS geotags describe synchronized
sensor centers from the same navigation solution. It cannot recover a sensor
offset absent from those original tags or fix an incorrectly paired PPK run.

Coordinates inherit the corrected RGB reference frame. The calculations use
ellipsoidal heights and WGS84 ellipsoid geometry for the short offset vector;
they are NOT a datum/epoch/geoid transformation. In Metashape, explicitly use
the reference CRS represented by your Emlid base coordinates, which might be
NAD83(2011). Do not infer that CRS from the GPSMapDatum tag alone.

OUTPUTS
* A separate drone_MS_PPK_corrected tree with flight/MS/original_name.TIF.
* Per-run audit CSV and summary JSON under _reports.
* A Green-band reference CSV, including full-filename and filename-stem
  labels so you can select the one matching Metashape's camera labels.

Originals and existing corrected-RGB/MS folders are not edited. Every output
is staged, its positions and retained XMP metadata are checked, and its image
data hash is compared with the original before publishing the file. Existing
outputs are checked and reused, or reported as conflicts; no overwrite by
default. Partial runs can be rerun safely using the same original inputs.

Quality tags: default 'clear' removes the TIFF's old RTK accuracy/fix tags.
Set camera reference accuracies in Metashape using your assessed PPK quality.
Optional 'copy_rgb' copies available RtkStd*/RtkFlag tags from the corrected
RGB ONLY; use it after confirming those tags describe the new PPK solution.
No accuracy or fixed-solution status is invented.

DRY_RUN=True creates reports only. The default False writes corrected copies.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ----------------------------- USER SETTINGS -----------------------------
ORIGINAL_RGB_ROOT = Path(r"E:\SfM\20261008_RC_imagery")
CORRECTED_RGB_ROOT = Path(
    r"C:\L\Lichen\Lichen - Documents\Projects\20240010_RoaringCr Des (CCD)"
    r"\06_Photos\20261009 - Drone flight\drone"
)
ORIGINAL_MS_ROOT = Path(r"E:\SfM\20261008_RC_imagery")

# Separate sibling of 'drone'; does not overwrite its existing MS subfolders.
OUTPUT_MS_ROOT = CORRECTED_RGB_ROOT.with_name("drone_MS_PPK_corrected")

# None searches beside this script and on PATH. Or enter a full executable path.
EXIFTOOL_PATH: str | None = None  # e.g. r"C:\Tools\ExifTool\exiftool.exe"
DRY_RUN = False
OVERWRITE_OUTPUTS = False

# 'auto' requires original EXIF/XMP positions to agree, when both are present.
# Explicit 'exif' or 'xmp' overrides that original-position consistency check.
ORIGINAL_POSITION_SOURCE = "auto"
# Emlid writes standard EXIF GPS positions; use them even if old XMP is stale.
CORRECTED_POSITION_SOURCE = "exif"
ACCURACY_MODE = "clear"  # 'clear' or 'copy_rgb'; see module documentation.

# Sanity limits, not accuracy estimates. Excesses are reported, not written.
MAX_RGB_CORRECTION_M = 250.0
MAX_ORIGINAL_SENSOR_OFFSET_M = 2.0
METADATA_CONFLICT_TOLERANCE_M = 0.20
OUTPUT_POSITION_TOLERANCE_M = 0.005
BATCH_SIZE = 40
# -------------------------------------------------------------------------

RGB_PATTERN = re.compile(r"^(DJI_\d{14}_\d+)_D\.(?:JPG|JPEG)$", re.I)
MS_PATTERN = re.compile(r"^(DJI_\d{14}_\d+)_MS_(G|R|RE|NIR)\.(?:TIF|TIFF)$", re.I)
WGS84_A = 6378137.0
WGS84_E2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)
QUALITY_TAGS = (
    "XMP-drone-dji:RtkStdLat", "XMP-drone-dji:RtkStdLon",
    "XMP-drone-dji:RtkStdHgt", "XMP-drone-dji:RtkFlag",
    "XMP-Camera:GPSXYAccuracy", "XMP-Camera:GPSZAccuracy",
)
POSITION_XMP_TAGS = {
    "XMP-drone-dji:GPSLatitude", "XMP-drone-dji:GPSLongitude",
    "XMP-drone-dji:GPSLongtitude", "XMP-drone-dji:Latitude",
    "XMP-drone-dji:Longitude", "XMP-drone-dji:AbsoluteAltitude",
    "XMP-exif:GPSLatitude", "XMP-exif:GPSLongitude",
    "XMP-exif:GPSAltitude", "XMP-exif:GPSAltitudeRef",
    "XMP-exif:GPSMapDatum",
    "XMP-x:XMPToolkit",  # Serialization software marker, not sensor metadata.
}


@dataclass(frozen=True)
class Position:
    latitude: float
    longitude: float
    height: float


@dataclass(frozen=True)
class Settings:
    original_rgb: Path = ORIGINAL_RGB_ROOT
    corrected_rgb: Path = CORRECTED_RGB_ROOT
    original_ms: Path = ORIGINAL_MS_ROOT
    output: Path = OUTPUT_MS_ROOT
    exiftool: str | None = EXIFTOOL_PATH
    dry_run: bool = DRY_RUN
    overwrite: bool = OVERWRITE_OUTPUTS
    original_source: str = ORIGINAL_POSITION_SOURCE
    corrected_source: str = CORRECTED_POSITION_SOURCE
    accuracy_mode: str = ACCURACY_MODE
    max_correction: float = MAX_RGB_CORRECTION_M
    max_offset: float = MAX_ORIGINAL_SENSOR_OFFSET_M
    conflict_tolerance: float = METADATA_CONFLICT_TOLERANCE_M
    batch_size: int = BATCH_SIZE


def path_key(path: Path | str) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def find_exiftool(configured: str | None) -> str:
    candidates = [configured] if configured else [
        str(Path(__file__).resolve().parent / "exiftool.exe"),
        shutil.which("exiftool"), shutil.which("exiftool.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            version = subprocess.run([candidate, "-ver"], capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", timeout=30)
            if version.returncode == 0:
                print(f"ExifTool {version.stdout.strip()}: {candidate}")
                return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "ExifTool was not found. Install its Windows distribution from "
        "https://exiftool.org/, retain the companion files, rename "
        "exiftool(-k).exe to exiftool.exe, and set EXIFTOOL_PATH or put the "
        "executable beside this script."
    )


def run_exiftool(executable: str, arguments: list[str]) -> subprocess.CompletedProcess:
    # An argument file avoids Windows command-line limits and preserves spaces.
    if any("\n" in arg or "\r" in arg for arg in arguments):
        raise ValueError("Newlines in ExifTool arguments/paths are unsupported.")
    with tempfile.TemporaryDirectory(prefix="m3m_args_") as temp:
        argfile = Path(temp) / "args.txt"
        argfile.write_text("\n".join(arguments) + "\n", encoding="utf-8")
        return subprocess.run(
            [executable, "-charset", "filename=UTF8", "-@", str(argfile)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=3600, shell=False,
        )


def read_metadata(executable: str, paths: list[Path], hashes: bool = False,
                  batch_size: int = BATCH_SIZE) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    unique = list(dict.fromkeys(Path(p).resolve() for p in paths))
    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        args = ["-j", "-n", "-G1", "-s", "-EXIF:all", "-XMP:all", "-Error", "-Warning"]
        if hashes:
            args += ["-api", "ImageHashType=SHA256", "-ImageDataHash"]
        proc = run_exiftool(executable, args + [str(p) for p in batch])
        try:
            records = json.loads(proc.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"ExifTool metadata read failed: {proc.stderr[:1500]}") from error
        for record in records:
            results[path_key(record["SourceFile"])] = record
        if proc.stderr.strip():
            print(f"  ExifTool read note: {proc.stderr.strip()[:600]}")
    return results


def geodetic_to_xyz(p: Position) -> tuple[float, float, float]:
    lat, lon = math.radians(p.latitude), math.radians(p.longitude)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    return ((n + p.height) * math.cos(lat) * math.cos(lon),
            (n + p.height) * math.cos(lat) * math.sin(lon),
            (n * (1 - WGS84_E2) + p.height) * math.sin(lat))


def xyz_to_geodetic(xyz: tuple[float, float, float]) -> Position:
    x, y, z = xyz
    rho = math.hypot(x, y)
    if rho < 1e-9:
        raise ValueError("Polar or invalid camera position is unsupported.")
    lat = math.atan2(z, rho * (1 - WGS84_E2))
    for _ in range(15):
        n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
        updated = math.atan2(z + WGS84_E2 * n * math.sin(lat), rho)
        if abs(updated - lat) < 1e-14:
            lat = updated
            break
        lat = updated
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    height = rho / math.cos(lat) - n
    return Position(math.degrees(lat), math.degrees(math.atan2(y, x)), height)


def distance(a: Position, b: Position) -> float:
    return math.dist(geodetic_to_xyz(a), geodetic_to_xyz(b))


def raw_position(meta: dict[str, Any], source: str) -> Position | None:
    if source == "exif":
        fields = [meta.get(f"GPS:{tag}") for tag in
                  ("GPSLatitude", "GPSLongitude", "GPSAltitude")]
        if any(value is None for value in fields):
            return None
        lat_ref = str(meta.get("GPS:GPSLatitudeRef", "")).upper()
        lon_ref = str(meta.get("GPS:GPSLongitudeRef", "")).upper()
        alt_ref = int(meta.get("GPS:GPSAltitudeRef", -1))
        if lat_ref not in ("N", "S") or lon_ref not in ("E", "W") or alt_ref not in (0, 1):
            raise ValueError("Missing/unsupported EXIF GPS hemisphere or altitude reference.")
        lat = abs(float(fields[0])) * (-1 if lat_ref == "S" else 1)
        lon = abs(float(fields[1])) * (-1 if lon_ref == "W" else 1)
        height = abs(float(fields[2])) * (-1 if alt_ref == 1 else 1)
    else:
        lat = meta.get("XMP-drone-dji:GPSLatitude", meta.get("XMP-drone-dji:Latitude"))
        lon = meta.get("XMP-drone-dji:GPSLongitude", meta.get(
            "XMP-drone-dji:GPSLongtitude", meta.get("XMP-drone-dji:Longitude")))
        height = meta.get("XMP-drone-dji:AbsoluteAltitude")
        if lat is None or lon is None or height is None:
            return None
        lat, lon, height = float(lat), float(lon), float(height)
    if not all(math.isfinite(v) for v in (lat, lon, height)):
        raise ValueError("Non-finite camera position.")
    if not -90 < lat < 90 or not -180 <= lon <= 180 or (lat == 0 and lon == 0):
        raise ValueError("Invalid or placeholder latitude/longitude.")
    return Position(lat, lon, height)


def select_position(meta: dict[str, Any], source: str, tolerance: float
                    ) -> tuple[Position, str, str]:
    error = meta.get("ExifTool:Error", meta.get("Error"))
    if error:
        raise ValueError(f"Metadata error: {error}")
    exif = raw_position(meta, "exif")
    xmp = raw_position(meta, "xmp")
    conflict = ""
    if exif is not None and xmp is not None:
        disagreement = distance(exif, xmp)
        if disagreement > tolerance:
            conflict = f"EXIF/XMP differ by {disagreement:.3f} m; selected {source}."
            if source == "auto":
                raise ValueError(conflict + " Inspect the original metadata before overriding the source.")
    chosen = (exif or xmp) if source == "auto" else (exif if source == "exif" else xmp)
    if chosen is None:
        raise ValueError(f"No complete {source} latitude/longitude/absolute-height position.")
    selected_source = "exif" if chosen is exif else "xmp"
    return chosen, selected_source, conflict


def make_index(paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[path.name.casefold()].append(path)
    return index


def choose_match(candidates: list[Path], preferred: Path) -> Path:
    if not candidates:
        raise ValueError(f"Missing matching RGB: {preferred.name}")
    preferred_key = path_key(preferred)
    exact = [p for p in candidates if path_key(p) == preferred_key]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Ambiguous RGB filename ({len(candidates)} matches): {preferred.name}")


def relative_flight_folder(ms_path: Path, root: Path) -> Path:
    parent = ms_path.relative_to(root).parent
    # Source TIFFs may be beside RGBs, or already inside a flight/MS folder.
    if parent.name.casefold() == "ms":
        parent = parent.parent
    return parent


def validate_settings(s: Settings) -> None:
    for root in (s.original_rgb, s.corrected_rgb, s.original_ms):
        if not root.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {root}")
    if path_key(s.original_rgb) == path_key(s.corrected_rgb):
        raise ValueError("Original and corrected RGB directories must differ.")
    for source in (s.original_rgb, s.original_ms, s.corrected_rgb):
        if s.output.resolve() == source.resolve() or s.output.resolve() in source.resolve().parents:
            raise ValueError("Output cannot equal or contain an input directory.")
    for source in (s.original_rgb, s.original_ms):
        if source.resolve() in s.output.resolve().parents:
            raise ValueError("Put output outside the original-image trees to prevent reprocessing outputs.")
    if s.original_source not in ("auto", "exif", "xmp") or s.corrected_source not in ("auto", "exif", "xmp"):
        raise ValueError("Position sources must be auto, exif, or xmp.")
    if s.accuracy_mode not in ("clear", "copy_rgb") or s.batch_size < 1:
        raise ValueError("Invalid accuracy mode or batch size.")


def plan_jobs(s: Settings, original_index: dict, corrected_index: dict,
              ms_files: list[Path]) -> list[dict[str, Any]]:
    jobs = []
    for ms in ms_files:
        match = MS_PATTERN.fullmatch(ms.name)
        assert match is not None
        capture, band = match.group(1), match.group(2).upper()
        row: dict[str, Any] = dict(status="pending", band=band, capture=capture,
                                   original_ms=str(ms), message="")
        job = {"row": row, "ms": ms}
        try:
            flight = relative_flight_folder(ms, s.original_ms)
            rgb_name = capture + "_D.JPG"
            candidates = list(original_index.get(rgb_name.casefold(), []))
            candidates += original_index.get((capture + "_D.JPEG").casefold(), [])
            original = choose_match(candidates, s.original_rgb / flight / rgb_name)
            corrected = choose_match(corrected_index.get(original.name.casefold(), []),
                                     s.corrected_rgb / original.relative_to(s.original_rgb))
            if path_key(original) == path_key(corrected):
                raise ValueError("Matched the same file as original and corrected RGB.")
            output = s.output / flight / "MS" / ms.name
            row.update(original_rgb=str(original), corrected_rgb=str(corrected), output_ms=str(output))
            job.update(original=original, corrected=corrected, output=output)
        except (ValueError, OSError) as error:
            row.update(status="match_error", message=str(error))
        jobs.append(job)
    counts = Counter(path_key(j["output"]) for j in jobs if "output" in j)
    for job in jobs:
        if "output" in job and counts[path_key(job["output"])] > 1:
            job["row"].update(status="match_error", message="Multiple sources would write the same output.")
    return jobs


def calculate_job(job: dict, metadata: dict, s: Settings) -> None:
    row = job["row"]
    ms_meta = metadata[path_key(job["ms"])]
    original_meta = metadata[path_key(job["original"])]
    corrected_meta = metadata[path_key(job["corrected"])]
    original, source_o, note_o = select_position(original_meta, s.original_source, s.conflict_tolerance)
    corrected, source_c, note_c = select_position(corrected_meta, s.corrected_source, s.conflict_tolerance)
    ms, source_m, note_m = select_position(ms_meta, s.original_source, s.conflict_tolerance)
    if source_m != source_o:
        raise ValueError("Original RGB/MS would use different position namespaces; select a common source explicitly.")
    correction = tuple(c - o for c, o in zip(geodetic_to_xyz(corrected), geodetic_to_xyz(original)))
    offset = tuple(m - o for m, o in zip(geodetic_to_xyz(ms), geodetic_to_xyz(original)))
    correction_length, offset_length = math.hypot(*correction), math.hypot(*offset)
    if correction_length > s.max_correction:
        raise ValueError(f"RGB correction {correction_length:.3f} m exceeds {s.max_correction} m.")
    if offset_length > s.max_offset:
        raise ValueError(f"Original MS-to-RGB offset {offset_length:.3f} m exceeds {s.max_offset} m.")
    result = xyz_to_geodetic(tuple(c + d for c, d in zip(geodetic_to_xyz(corrected), offset)))
    notes = [note for note in (note_o, note_c, note_m) if note]
    if correction_length < 0.001:
        notes.append("RGB positions differ by less than 1 mm; confirm these JPGs were corrected.")
    if offset_length < 0.001:
        notes.append("Original geotags contain no resolvable MS-to-RGB offset.")
    datum = corrected_meta.get("GPS:GPSMapDatum")
    row.update(
        status="planned", message=" ".join(notes), original_position_source=source_o,
        corrected_position_source=source_c, corrected_rgb_map_datum_tag=datum or "",
        original_ms_latitude=ms.latitude, original_ms_longitude=ms.longitude,
        original_ms_height_m=ms.height, corrected_latitude=result.latitude,
        corrected_longitude=result.longitude, corrected_height_m=result.height,
        rgb_correction_m=correction_length, original_sensor_offset_m=offset_length,
        correction_x_m=correction[0], correction_y_m=correction[1], correction_z_m=correction[2],
        offset_x_m=offset[0], offset_y_m=offset[1], offset_z_m=offset[2],
    )
    job.update(position=result, ms_meta=ms_meta, corrected_meta=corrected_meta)


def write_arguments(job: dict, target: Path, s: Settings) -> list[str]:
    p, original, corrected = job["position"], job["ms_meta"], job["corrected_meta"]
    tags: dict[str, Any] = {
        "GPS:GPSLatitude": abs(p.latitude), "GPS:GPSLatitudeRef": "N" if p.latitude >= 0 else "S",
        "GPS:GPSLongitude": abs(p.longitude), "GPS:GPSLongitudeRef": "E" if p.longitude >= 0 else "W",
        "GPS:GPSAltitude": abs(p.height), "GPS:GPSAltitudeRef": 0 if p.height >= 0 else 1,
        "XMP-drone-dji:GPSLatitude": p.latitude, "XMP-drone-dji:GPSLongitude": p.longitude,
        "XMP-drone-dji:AbsoluteAltitude": p.height,
    }
    for tag, value in (
        ("XMP-drone-dji:Latitude", p.latitude), ("XMP-drone-dji:Longitude", p.longitude),
        ("XMP-drone-dji:GPSLongtitude", p.longitude),
        ("XMP-exif:GPSLatitude", p.latitude), ("XMP-exif:GPSLongitude", p.longitude),
        ("XMP-exif:GPSAltitude", abs(p.height)), ("XMP-exif:GPSAltitudeRef", int(p.height < 0)),
    ):
        if tag in original:
            tags[tag] = value
    if corrected.get("GPS:GPSMapDatum"):
        tags["GPS:GPSMapDatum"] = corrected["GPS:GPSMapDatum"]
        if "XMP-exif:GPSMapDatum" in original:
            tags["XMP-exif:GPSMapDatum"] = corrected["GPS:GPSMapDatum"]
    for tag in QUALITY_TAGS:
        tags[tag] = ""
        if s.accuracy_mode == "copy_rgb" and corrected.get(tag) is not None:
            value = corrected[tag]
            if tag.endswith("RtkFlag"):
                tags[tag] = value
            elif math.isfinite(float(value)) and float(value) > 0:
                tags[tag] = value
    arguments = ["-overwrite_original", "-P", "-n", "-charset", "filename=UTF8"]
    for tag, value in tags.items():
        formatted = format(value, ".12f") if isinstance(value, float) else str(value)
        arguments.append(f"-{tag}={formatted}")
    return arguments + [str(target.resolve())]


def verify_output(job: dict, after: dict, s: Settings) -> None:
    before = job["ms_meta"]
    expected = job["position"]
    for source in ("exif", "xmp"):
        actual = raw_position(after, source)
        if actual is None or distance(actual, expected) > OUTPUT_POSITION_TOLERANCE_M:
            raise ValueError(f"Written {source} position differs from the expected position.")
    old_hash, new_hash = before.get("File:ImageDataHash"), after.get("File:ImageDataHash")
    if not old_hash or not new_hash:
        raise ValueError("ExifTool did not return an image-data hash; output was not accepted.")
    if old_hash != new_hash:
        raise ValueError("Image-data hash changed; output was not accepted.")
    allowed = POSITION_XMP_TAGS | set(QUALITY_TAGS)
    for tag, value in before.items():
        if tag.startswith("XMP-") and tag not in allowed and after.get(tag) != value:
            raise ValueError(f"Retained XMP metadata changed or disappeared: {tag}")
    for tag in ("IFD0:ImageWidth", "IFD0:ImageHeight", "IFD0:BitsPerSample",
                "IFD0:Compression", "IFD0:SamplesPerPixel", "ExifIFD:DateTimeOriginal"):
        if tag in before and after.get(tag) != before[tag]:
            raise ValueError(f"TIFF structure/capture metadata changed: {tag}")
    if s.accuracy_mode == "clear" and any(tag in after for tag in QUALITY_TAGS):
        raise ValueError("Old RTK accuracy/fix tags were not removed.")


def process_batch(batch: list[dict], executable: str, s: Settings) -> None:
    staged_jobs = []
    with tempfile.TemporaryDirectory(prefix=".m3m_stage_", dir=s.output) as staging:
        arguments = []
        for index, job in enumerate(batch):
            try:
                staged = Path(staging) / f"{index:05d}.TIF"
                shutil.copy2(job["ms"], staged)
                staged.chmod(staged.stat().st_mode | 0o200)
                if arguments:
                    arguments.append("-execute")
                arguments.extend(write_arguments(job, staged, s))
                staged_jobs.append((job, staged))
            except (OSError, ValueError) as error:
                job["row"].update(status="write_error", message=str(error))
        if not staged_jobs:
            return
        proc = run_exiftool(executable, arguments)
        if proc.stderr.strip():
            print(f"  ExifTool write note: {proc.stderr.strip()[:800]}")
        after_all = read_metadata(executable, [p for _, p in staged_jobs], hashes=True,
                                  batch_size=s.batch_size)
        for job, staged in staged_jobs:
            try:
                verify_output(job, after_all[path_key(staged)], s)
                output = job["output"]
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists() and not s.overwrite:
                    raise FileExistsError("Output appeared during processing; rerun to verify it.")
                # Stage and output share a filesystem; replace is atomic.
                os.replace(staged, output)
                job["row"]["status"] = "written"
            except (OSError, ValueError, KeyError) as error:
                job["row"].update(status="write_error", message=str(error))


def save_reports(jobs: list[dict], s: Settings, executable: str, run_id: str) -> dict:
    reports = s.output / "_reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = [job["row"] for job in jobs]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    audit = reports / f"correction_audit_{run_id}.csv"
    with audit.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    references = reports / f"metashape_reference_G_{run_id}.csv"
    with references.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "label_stem", "longitude", "latitude", "height_m", "output_file"])
        for job in jobs:
            row = job["row"]
            if row["band"] == "G" and row["status"] in ("written", "existing_verified"):
                p = job["position"]
                writer.writerow([job["ms"].name, job["ms"].stem,
                                 f"{p.longitude:.12f}", f"{p.latitude:.12f}",
                                 f"{p.height:.6f}", str(job["output"])])
    counts = dict(Counter(row["status"] for row in rows))
    summary = {
        "run_utc": run_id, "status_counts": counts, "dry_run": s.dry_run,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(s).items()},
        "exiftool": executable, "audit_csv": str(audit), "green_reference_csv": str(references),
        "method": "MS_corrected_XYZ = RGB_corrected_XYZ + MS_original_XYZ - RGB_original_XYZ",
        "coordinate_reference": "Inherited from corrected RGB; no datum/epoch/geoid transformation.",
        "height_assumption": "Absolute ellipsoidal heights; never takeoff-relative heights.",
        "quality_note": "No independent PPK quality assessment; review Emlid solution and checkpoints.",
    }
    shifts = [row["rgb_correction_m"] for row in rows if "rgb_correction_m" in row]
    if shifts:
        summary["rgb_correction_range_m"] = [min(shifts), max(shifts)]
    (reports / f"summary_{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nStatus counts:", counts)
    print("Audit:", audit)
    print("Green reference CSV:", references)
    print("Output TIFF root:", s.output)
    return summary


def run(settings: Settings | None = None) -> dict:
    s = settings or Settings()
    validate_settings(s)
    executable = find_exiftool(s.exiftool)
    print("Indexing filenames recursively...")
    original_rgbs = sorted(p for p in s.original_rgb.rglob("*") if p.is_file() and RGB_PATTERN.fullmatch(p.name))
    corrected_rgbs = sorted(p for p in s.corrected_rgb.rglob("*") if p.is_file() and RGB_PATTERN.fullmatch(p.name))
    ms_files = sorted(p for p in s.original_ms.rglob("*") if p.is_file() and MS_PATTERN.fullmatch(p.name))
    if not ms_files:
        raise FileNotFoundError("No DJI *_MS_G/R/RE/NIR.TIF files found in ORIGINAL_MS_ROOT.")
    print(f"Found {len(original_rgbs)} original RGB, {len(corrected_rgbs)} corrected RGB, {len(ms_files)} MS files.")
    jobs = plan_jobs(s, make_index(original_rgbs), make_index(corrected_rgbs), ms_files)
    eligible = [j for j in jobs if j["row"]["status"] == "pending"]
    rgb_paths = [j[k] for j in eligible for k in ("original", "corrected")]
    print("Reading paired RGB positions...")
    metadata = read_metadata(executable, rgb_paths, batch_size=s.batch_size)
    print("Reading original TIFF metadata and image-data hashes...")
    metadata.update(read_metadata(executable, [j["ms"] for j in eligible],
                                  hashes=True, batch_size=s.batch_size))
    for job in eligible:
        try:
            calculate_job(job, metadata, s)
        except (ValueError, KeyError, TypeError, OSError) as error:
            job["row"].update(status="position_error", message=str(error))
    planned = [j for j in jobs if j["row"]["status"] == "planned"]
    s.output.mkdir(parents=True, exist_ok=True)
    if s.dry_run:
        for job in planned:
            job["row"]["status"] = "dry_run"
    else:
        existing = [j for j in planned if j["output"].exists() and not s.overwrite]
        existing_meta = read_metadata(executable, [j["output"] for j in existing],
                                      hashes=True, batch_size=s.batch_size)
        for job in existing:
            try:
                verify_output(job, existing_meta[path_key(job["output"])], s)
                job["row"]["status"] = "existing_verified"
            except (ValueError, KeyError, TypeError) as error:
                job["row"].update(status="existing_conflict", message=str(error))
        pending = [j for j in planned if j["row"]["status"] == "planned"]
        for start in range(0, len(pending), s.batch_size):
            batch = pending[start:start + s.batch_size]
            print(f"Writing/checking TIFFs {start + 1}-{start + len(batch)} of {len(pending)}...")
            try:
                process_batch(batch, executable, s)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                for job in batch:
                    if job["row"]["status"] == "planned":
                        job["row"].update(status="write_error", message=str(error))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return save_reports(jobs, s, executable, run_id)


if __name__ == "__main__":
    result = run()
    failures = sum(count for status, count in result["status_counts"].items()
                   if status.endswith("_error") or status == "existing_conflict")
    if failures:
        print(f"\n{failures} file(s) need attention; see the audit CSV. Valid files were processed.")
        raise SystemExit(1)
