"""Object-based multispectral river-cover classification with Random Forest.

RUN IN VSCODE
1. Edit the SETTINGS block below. Confirm the multispectral band order.
2. Run this file. If the training GeoPackage does not exist, an empty training
   layer is created. Draw polygons in QGIS, populate class_id, save, and rerun.
3. Inspect the held-out validation results, uncertainty, and wood/foam polygons.

TRAINING POLYGON FIELDS
  class_id : integer from CLASS_NAMES; one surface-cover class per polygon.
  split    : 'train' or 'validate' for a manual holdout; leave ALL values blank
             for automatic spatial-group splitting. 'ignore' is also supported.
  group_id : optional spatial sampling group. Otherwise polygon centroids are
             grouped into blocks of VALIDATION_BLOCK_SIZE_M. Entire polygons
             remain together. Draw polygons throughout the mapped corridor.
Training polygons must not overlap. Use enough geographically separated
examples of each class for both training and validation. Training polygons
close to validation polygons are excluded by VALIDATION_BUFFER_M.

INPUTS
Required: four-band green/red/red-edge/NIR imagery (or four individual rasters),
and labeled training polygons. Optional: RGB, DEM, HAWS REM, DSM + bare-earth
DTM, and AOI polygons. A projected CRS is required; horizontal metre and foot
units are handled. Set terrain vertical conversion factors explicitly.
Reflectance-calibrated imagery is preferred. RGB should be unstretched source
imagery. Index formulas assume a common, correctly specified MS scale/offset.
DSM and DTM must share vertical datum and compatible acquisition dates.

OUTPUTS (a new timestamped run directory, never overwriting a previous run)
  classification.tif         accepted class, 254 = uncertain, 0 = no data
  predicted_classes.tif      original RF class before uncertainty rejection
  confidence.tif             largest RF class score, -9999 = no data
  score_margin.tif           difference between two largest class scores
  class_scores.tif           optional score band for each trained class
  visible_water.tif          1 = water/whitewater, 0 = other, 255 = unknown
  segments.tif               unique image-object IDs; 0 = no data
  classified_objects.gpkg    optional individual segmented object polygons
  training_polygons_used.gpkg original polygons with resolved split/group
  object_predictions.csv     object IDs, classes, scores, reference information
  validation_*.csv           object and reference-pixel confusion/metrics
  mapped_area_summary.csv    mapped surface area, not error-adjusted estimates
  feature_importance.csv     RF impurity importance, for exploratory review
  trained_model.joblib       imputer, forest, feature order, settings
  run_report.json            settings, counts, validation, and limitations

METHOD / LIMITS
SLIC creates image objects in tiles with contextual halos. Objects stop at tile
boundaries; the halo reduces texture edge effects but does not remove segment
seams. Select an analysis resolution and segment spacing that retain wood.
Terrain is ancillary, never a hard water mask. Whitewater is visible foam and
is grouped with water only in visible_water.tif. Visible wood cover does not
measure hidden wood, individual-piece counts, or wood volume.
RF scores are uncalibrated class scores, not calibrated accuracy probabilities.
Validation uses untouched polygons/groups and a training exclusion buffer.
Reference-pixel metrics use every valid imagery pixel in validation polygons;
object metrics use only sufficiently covered, sufficiently pure objects.
The final model uses training data only, so validation remains held out.
Use a small representative AOI first; inspect segment boundaries before scaling.

DEPENDENCIES: Python >= 3.10, rasterio, geopandas, shapely, pyproj, numpy,
pandas, scipy, scikit-image >= 0.20, scikit-learn >= 1.2, joblib, pyogrio.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.metadata
import json
import math
import time

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import CRS as PyprojCRS
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds
from rasterio.windows import Window
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
from shapely.geometry import box, shape
from shapely.ops import unary_union
from skimage.measure import label as connected_labels, regionprops_table
from skimage.segmentation import slic
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


# ================================ SETTINGS ================================
# Use VSCode's Run Python File button after editing these settings.
PROJECT_DIR = Path(r"C:\GIS\River_Classification")
MULTISPECTRAL_PATH = PROJECT_DIR / "multispectral_orthomosaic.tif"
# Do not assume band order. Confirm it in the source file or export settings.
MS_BANDS = {"green": 1, "red": 2, "red_edge": 3, "nir": 4}
# To use individual band files instead, provide ALL four names here:
# MS_BAND_FILES = {
#     "green": (PROJECT_DIR / "green.tif", 1),
#     "red": (PROJECT_DIR / "red.tif", 1),
#     "red_edge": (PROJECT_DIR / "red_edge.tif", 1),
#     "nir": (PROJECT_DIR / "nir.tif", 1),
# }
MS_BAND_FILES = {}
MS_SCALE_FACTOR = 1.0           # Example: 1 / 32768 ONLY if correct for export.
MS_OFFSET = 0.0                 # Scaled value = stored value * scale + offset.

# Set optional paths to None when unavailable. Configured paths must exist.
RGB_PATH = None                 # Example: PROJECT_DIR / "rgb_orthomosaic.tif"
RGB_BANDS = {"rgb_red": 1, "rgb_green": 2, "rgb_blue": 3}
RGB_SCALE_FACTOR = None         # None: integer / dtype maximum; float unchanged.
DEM_PATH = None                 # Bare-earth terrain OR SfM surface; identify it.
DEM_Z_TO_METRES = 1.0           # Set 0.3048 for international feet.
REM_PATH = None                 # HAWS REM; continuous height, not categorized.
REM_Z_TO_METRES = 1.0           # Set 0.3048 if REM heights are in feet.
DSM_PATH = None                 # Both DSM and DTM are needed for height features.
DTM_PATH = None
DSM_Z_TO_METRES = 1.0
DTM_Z_TO_METRES = 1.0
AOI_PATH = None                 # Optional polygon GeoPackage/shapefile.
AOI_LAYER = None

TRAINING_PATH = PROJECT_DIR / "river_training.gpkg"
TRAINING_LAYER = "training"
CLASS_FIELD = "class_id"
SPLIT_FIELD = "split"
GROUP_FIELD = "group_id"
CREATE_TRAINING_TEMPLATE_IF_MISSING = True
CLASS_NAMES = {
    1: "Exposed wood", 2: "Vegetation", 3: "Open gravel",
    4: "Water without visible foam", 5: "Whitewater / foam",
    6: "Obscured", 7: "Bare soil", 8: "Bedrock / boulders",
}
# Remove a class from this tuple if it is genuinely absent from the study site.
# Optional classes 6-8 are trained only if you provide polygons for them.
REQUIRED_CLASS_IDS = (1, 2, 3, 4, 5)
CLASS_COLORS = {
    0: (0, 0, 0, 0), 1: (139, 92, 49, 255), 2: (58, 133, 62, 255),
    3: (205, 188, 154, 255), 4: (36, 114, 177, 255),
    5: (168, 224, 244, 255), 6: (82, 82, 82, 255),
    7: (167, 128, 92, 255), 8: (157, 157, 157, 255),
    254: (216, 79, 151, 255),
}

ANALYSIS_CRS = None             # None: reference imagery CRS; e.g. "EPSG:32611".
ANALYSIS_RESOLUTION_M = None    # None: native/reference resolution. E.g. 0.05.
TILE_SIZE_PIXELS = 1024
CONTEXT_HALO_PIXELS = 32        # Increased automatically for texture kernels.
SEGMENT_SPACING_M = 0.30        # Nominal spacing, not a minimum feature width.
SLIC_COMPACTNESS = 0.05         # Lower values follow spectral edges more closely.
SLIC_SIGMA_PIXELS = 0.0         # Smoothing can erase narrow logs.
SLIC_ITERATIONS = 10
TEXTURE_WINDOW_DIAMETERS_M = (0.15, 0.50, 1.50)
TERRAIN_RELIEF_WINDOW_M = 2.0

VALIDATION_FRACTION = 0.25
VALIDATION_BLOCK_SIZE_M = 50.0
VALIDATION_BUFFER_M = 10.0      # Distance from validation polygons, not centroids.
REQUIRE_ALL_CLASSES_IN_VALIDATION = True
MIN_TRAINING_OBJECTS_PER_CLASS = 5
MIN_POLYGON_COVERAGE = 0.50     # Fraction of object covered by reference polygons.
MIN_LABEL_PURITY = 0.95        # Same-class fraction of labeled object pixels.

RF_TREES = 300
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = "sqrt"
RF_N_JOBS = -1
MAX_TRAINING_OBJECTS_PER_CLASS = 20000
RANDOM_SEED = 42
MIN_CLASS_SCORE = 0.60          # Starting threshold; select using validation.
MIN_SCORE_MARGIN = 0.10
UNCERTAIN_CLASS_ID = 254
WATER_CLASS_IDS = (4, 5)
OBSCURED_CLASS_IDS = (6,)

OUTPUT_ROOT = PROJECT_DIR / "classification_output"
WRITE_CLASS_SCORE_RASTER = True
EXPORT_OBJECT_POLYGONS = True  # Can create a large GPKG at fine resolution.
# ========================================================================


@dataclass(frozen=True)
class Grid:
    crs: rasterio.crs.CRS
    transform: Affine
    width: int
    height: int
    unit_to_m: float
    resolution_m: float
    aoi: object | None


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def require_path(path: Path | str, description: str) -> Path:
    result = Path(path).expanduser()
    if not result.is_file():
        raise FileNotFoundError(f"{description} does not exist: {result}")
    return result


def check_settings() -> None:
    if not 0 < VALIDATION_FRACTION < 1:
        raise ValueError("VALIDATION_FRACTION must be between 0 and 1.")
    if not 0 < MIN_POLYGON_COVERAGE <= 1 or not 0 < MIN_LABEL_PURITY <= 1:
        raise ValueError("Training coverage and purity must be in (0, 1].")
    if (DSM_PATH is None) != (DTM_PATH is None):
        raise ValueError("Set both DSM_PATH and DTM_PATH, or leave both None.")
    if TILE_SIZE_PIXELS < 32 or SEGMENT_SPACING_M <= 0:
        raise ValueError("Use TILE_SIZE_PIXELS >= 32 and SEGMENT_SPACING_M > 0.")
    if VALIDATION_BUFFER_M < 0 or VALIDATION_BLOCK_SIZE_M <= 0:
        raise ValueError("Validation buffer must be nonnegative; block size positive.")
    if not 0 <= MIN_CLASS_SCORE <= 1 or not 0 <= MIN_SCORE_MARGIN <= 1:
        raise ValueError("Score thresholds must be between 0 and 1.")
    if any(c <= 0 or c >= 254 for c in CLASS_NAMES):
        raise ValueError("Training class IDs must be integers between 1 and 253.")
    if UNCERTAIN_CLASS_ID != 254:
        raise ValueError("This output schema reserves class 254 for uncertainty.")
    if any(c not in CLASS_NAMES for c in REQUIRED_CLASS_IDS):
        raise ValueError("REQUIRED_CLASS_IDS must be defined in CLASS_NAMES.")
    if any(d <= 0 for d in TEXTURE_WINDOW_DIAMETERS_M) or TERRAIN_RELIEF_WINDOW_M <= 0:
        raise ValueError("Texture and terrain window sizes must be positive.")
    if any(v <= 0 for v in (MS_SCALE_FACTOR, DEM_Z_TO_METRES, REM_Z_TO_METRES,
                           DSM_Z_TO_METRES, DTM_Z_TO_METRES)):
        raise ValueError("Radiometric and vertical scale factors must be positive.")
    if RGB_PATH is not None and set(RGB_BANDS) != {"rgb_red", "rgb_green", "rgb_blue"}:
        raise ValueError("RGB_BANDS must define rgb_red, rgb_green, and rgb_blue.")
    if MIN_TRAINING_OBJECTS_PER_CLASS < 1 or RF_TREES < 1 or RF_MIN_SAMPLES_LEAF < 1:
        raise ValueError("Training object count, RF_TREES, and RF_MIN_SAMPLES_LEAF must be positive.")
    if MAX_TRAINING_OBJECTS_PER_CLASS is not None and MAX_TRAINING_OBJECTS_PER_CLASS < 0:
        raise ValueError("MAX_TRAINING_OBJECTS_PER_CLASS must be nonnegative or None.")


def source_specs() -> dict[str, tuple[Path, int, float, float]]:
    specs = {}
    for name in ("green", "red", "red_edge", "nir"):
        if MS_BAND_FILES:
            if name not in MS_BAND_FILES:
                raise ValueError("MS_BAND_FILES must contain all four band names.")
            path, band = MS_BAND_FILES[name]
        else:
            path, band = MULTISPECTRAL_PATH, MS_BANDS[name]
        specs[name] = (require_path(path, name), int(band), MS_SCALE_FACTOR, MS_OFFSET)
    if RGB_PATH is not None:
        path = require_path(RGB_PATH, "RGB imagery")
        with rasterio.open(path) as source:
            dtype = np.dtype(source.dtypes[0])
            factor = RGB_SCALE_FACTOR
            if factor is None:
                factor = 1.0 / np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
        for name, band in RGB_BANDS.items():
            specs[name] = (path, int(band), float(factor), 0.0)
    for name, path, factor in (
        ("dem", DEM_PATH, DEM_Z_TO_METRES), ("rem", REM_PATH, REM_Z_TO_METRES),
        ("dsm", DSM_PATH, DSM_Z_TO_METRES), ("dtm", DTM_PATH, DTM_Z_TO_METRES),
    ):
        if path is not None:
            specs[name] = (require_path(path, name), 1, factor, 0.0)
    return specs


def read_polygon_file(path: Path | str, layer: str | None, crs) -> gpd.GeoDataFrame:
    kwargs = {"layer": layer} if layer is not None else {}
    gdf = gpd.read_file(require_path(path, "Polygon input"), **kwargs)
    if gdf.crs is None:
        raise ValueError(f"Polygon file has no CRS: {path}")
    if gdf.empty:
        raise ValueError(f"Polygon layer is empty: {path}")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError(f"Polygon layer contains empty geometry: {path}")
    if not gdf.geometry.is_valid.all():
        raise ValueError(f"Fix invalid polygon geometry in QGIS first: {path}")
    if not gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"Expected Polygon or MultiPolygon geometries: {path}")
    return gdf.to_crs(crs).reset_index(drop=True)


def make_grid(reference_path: Path) -> Grid:
    with rasterio.open(reference_path) as source:
        if source.crs is None:
            raise ValueError("Reference imagery has no CRS.")
        crs = rasterio.crs.CRS.from_user_input(ANALYSIS_CRS or source.crs)
        pcrs = PyprojCRS.from_user_input(crs)
        if not pcrs.is_projected:
            raise ValueError("Set ANALYSIS_CRS to a suitable projected CRS, such as local UTM.")
        axes = pcrs.axis_info[:2]
        if len(axes) != 2 or not np.isclose(axes[0].unit_conversion_factor, axes[1].unit_conversion_factor):
            raise ValueError("Analysis CRS must have matching linear units on both axes.")
        unit_to_m = float(axes[0].unit_conversion_factor)
        default_transform, _, _ = calculate_default_transform(
            source.crs, crs, source.width, source.height, *source.bounds)
        default_resolution = max(abs(default_transform.a), abs(default_transform.e)) * unit_to_m
        resolution_m = default_resolution if ANALYSIS_RESOLUTION_M is None else float(ANALYSIS_RESOLUTION_M)
        if resolution_m <= 0:
            raise ValueError("ANALYSIS_RESOLUTION_M must be positive.")
        bounds = transform_bounds(source.crs, crs, *source.bounds, densify_pts=21)
    aoi = None
    if AOI_PATH is not None:
        aoi = unary_union(read_polygon_file(AOI_PATH, AOI_LAYER, crs).geometry)
        footprint = aoi.intersection(box(*bounds))
        if footprint.is_empty or footprint.area == 0:
            raise ValueError("AOI does not intersect the multispectral imagery.")
        bounds = footprint.bounds
        aoi = footprint
    pixel = resolution_m / unit_to_m
    left, bottom, right, top = bounds
    # Keep the reference grid's phase. Relative coordinates avoid floating-point
    # roundoff expanding an otherwise aligned native grid by a pixel.
    anchor_x, anchor_y = default_transform.c, default_transform.f
    epsilon = 1e-7
    first_col = math.floor((left - anchor_x) / pixel + epsilon)
    last_col = math.ceil((right - anchor_x) / pixel - epsilon)
    first_row = math.floor((anchor_y - top) / pixel + epsilon)
    last_row = math.ceil((anchor_y - bottom) / pixel - epsilon)
    left, top = anchor_x + first_col * pixel, anchor_y - first_row * pixel
    width, height = last_col - first_col, last_row - first_row
    return Grid(crs, Affine(pixel, 0, left, 0, -pixel, top), width, height,
                unit_to_m, resolution_m, aoi)


def create_training_template(path: Path, grid: Grid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    empty = gpd.GeoDataFrame(
        {CLASS_FIELD: pd.Series(dtype="int32"), SPLIT_FIELD: pd.Series(dtype="str"),
         GROUP_FIELD: pd.Series(dtype="str")}, geometry=gpd.GeoSeries([], crs=grid.crs))
    empty.to_file(path, layer=TRAINING_LAYER, driver="GPKG", engine="pyogrio", geometry_type="Polygon")
    log(f"Created empty training layer: {path} | layer: {TRAINING_LAYER}")
    log("Draw polygons in QGIS, assign class_id, and save. Leave split blank for automatic grouping.")
    for class_id, name in CLASS_NAMES.items():
        log(f"  {class_id}: {name}")


def load_references(grid: Grid) -> tuple[gpd.GeoDataFrame, object, str]:
    gdf = read_polygon_file(TRAINING_PATH, TRAINING_LAYER, grid.crs)
    if CLASS_FIELD not in gdf:
        raise ValueError(f"Training layer needs an integer '{CLASS_FIELD}' field.")
    ids = pd.to_numeric(gdf[CLASS_FIELD], errors="coerce")
    if ids.isna().any() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("Every training polygon needs a whole-number class_id.")
    gdf[CLASS_FIELD] = ids.astype("int32")
    if not gdf[CLASS_FIELD].isin(CLASS_NAMES).all():
        raise ValueError("Unknown class_id in training polygons. Edit CLASS_NAMES if needed.")
    bounds = rasterio.transform.array_bounds(grid.height, grid.width, grid.transform)
    footprint = grid.aoi if grid.aoi is not None else box(*bounds)
    gdf.geometry = gdf.geometry.intersection(footprint)
    gdf = gdf.loc[~gdf.geometry.is_empty & (gdf.geometry.area > 0)].reset_index(drop=True)
    if gdf.empty:
        raise ValueError("No training polygons intersect the analysis footprint.")
    # Reject actual overlap, including same-class polygons, so reference areas are counted once.
    index = gdf.sindex
    for i, geom in enumerate(gdf.geometry):
        for j in index.query(geom, predicate="intersects"):
            if j > i and geom.intersection(gdf.geometry.iloc[j]).area > (grid.resolution_m / grid.unit_to_m) ** 2 * 1e-6:
                raise ValueError(f"Training polygons {i + 1} and {j + 1} overlap. Fix in QGIS.")
    roles = gdf[SPLIT_FIELD].fillna("").astype(str).str.strip().str.lower() if SPLIT_FIELD in gdf else pd.Series("", index=gdf.index)
    roles = roles.replace({"validation": "validate", "test": "validate", "training": "train"})
    if not roles.isin(["", "train", "validate", "ignore"]).all():
        raise ValueError("split must be blank, 'train', 'validate', or 'ignore'.")
    supplied_groups = gdf[GROUP_FIELD].fillna("").astype(str).str.strip() if GROUP_FIELD in gdf else pd.Series("", index=gdf.index)
    centroids = gdf.geometry.centroid
    blocks = [f"block_{math.floor(x * grid.unit_to_m / VALIDATION_BLOCK_SIZE_M)}_{math.floor(y * grid.unit_to_m / VALIDATION_BLOCK_SIZE_M)}"
              for x, y in zip(centroids.x, centroids.y)]
    groups = np.array([f"user_{v}" if v else b for v, b in zip(supplied_groups, blocks)], dtype=str)
    active = roles.ne("ignore").to_numpy()
    classes = set(gdf.loc[active, CLASS_FIELD])
    missing = set(REQUIRED_CLASS_IDS) - classes
    if missing:
        raise ValueError(f"No polygons for required classes {sorted(missing)}. Label them or adjust REQUIRED_CLASS_IDS.")
    automatic = not roles.loc[active].ne("").any()
    if automatic:
        positions = np.flatnonzero(active)
        if np.unique(groups[active]).size < 2:
            raise ValueError("Need multiple spatial groups. Add separated polygons, reduce block size, or set manual split.")
        splitter = GroupShuffleSplit(n_splits=500, test_size=VALIDATION_FRACTION, random_state=RANDOM_SEED)
        chosen = None
        for train_i, val_i in splitter.split(positions, groups=groups[active]):
            train_positions, val_positions = positions[train_i], positions[val_i]
            if set(gdf.iloc[train_positions][CLASS_FIELD]) != classes:
                continue
            if REQUIRE_ALL_CLASSES_IN_VALIDATION and set(gdf.iloc[val_positions][CLASS_FIELD]) != classes:
                continue
            chosen = (train_positions, val_positions)
            break
        if chosen is None:
            raise ValueError("Cannot hold out spatial groups with all classes represented. Add separated examples, use manual split, or relax REQUIRE_ALL_CLASSES_IN_VALIDATION.")
        roles.iloc[chosen[0]], roles.iloc[chosen[1]] = "train", "validate"
        method = "whole polygons grouped by group_id or centroid blocks"
    else:
        if roles.loc[active].eq("").any():
            raise ValueError("For manual splitting, fill split for every active polygon.")
        for group in supplied_groups[supplied_groups.ne("")].unique():
            if roles[(supplied_groups == group) & active].nunique() > 1:
                raise ValueError(f"Spatial group '{group}' occurs in both train and validate.")
        method = "user-specified whole-polygon split"
    if not roles.eq("train").any() or not roles.eq("validate").any():
        raise ValueError("Both train and validate polygons are required.")
    if set(gdf.loc[roles.eq("train"), CLASS_FIELD]) != classes:
        raise ValueError("Training polygons must represent every active class.")
    if REQUIRE_ALL_CLASSES_IN_VALIDATION and set(gdf.loc[roles.eq("validate"), CLASS_FIELD]) != classes:
        raise ValueError("Validation polygons must represent every active class, or relax REQUIRE_ALL_CLASSES_IN_VALIDATION.")
    gdf[SPLIT_FIELD] = roles
    gdf[GROUP_FIELD] = groups
    gdf["reference_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    gdf["role_code"] = roles.map({"train": 1, "validate": 2, "ignore": 3}).astype("int32")
    gdf["group_code"] = pd.factorize(groups)[0].astype("int32") + 1
    exclusion = unary_union(gdf.loc[roles.eq("validate"), "geometry"]).buffer(VALIDATION_BUFFER_M / grid.unit_to_m)
    return gdf, exclusion, method


def open_aligned_sources(specs, grid: Grid, stack: ExitStack):
    opened, readers = {}, {}
    for name, (path, band, gain, offset) in specs.items():
        if path not in opened:
            source = stack.enter_context(rasterio.open(path))
            if source.crs is None:
                raise ValueError(f"Raster has no CRS: {path}")
            vrt = stack.enter_context(WarpedVRT(
                source, crs=grid.crs, transform=grid.transform, width=grid.width,
                height=grid.height, dtype="float32", nodata=np.nan,
                resampling=Resampling.bilinear))
            opened[path] = (source, vrt)
        source, vrt = opened[path]
        if band < 1 or band > source.count:
            raise ValueError(f"Band {band} does not exist for {name}: {path}")
        readers[name] = (vrt, band, gain, offset)
    return readers


def read_layers(readers, window: Window | None = None, out_shape=None):
    result = {}
    for name, (source, band, gain, offset) in readers.items():
        kwargs = {"masked": True}
        if window is not None:
            kwargs["window"] = window
        if out_shape is not None:
            kwargs["out_shape"] = out_shape
            kwargs["resampling"] = Resampling.nearest
        arr = source.read(band, **kwargs).astype("float32").filled(np.nan)
        result[name] = arr * gain + offset
    return result


def find_normalizers(readers, grid: Grid):
    scale = min(1.0, 1024 / max(grid.width, grid.height))
    h, w = max(1, round(grid.height * scale)), max(1, round(grid.width * scale))
    layers = read_layers(readers, out_shape=(h, w))
    mask = np.ones((h, w), dtype=bool)
    for name in ("green", "red", "red_edge", "nir"):
        mask &= np.isfinite(layers[name])
    if grid.aoi is not None:
        transform = grid.transform * Affine.scale(grid.width / w, grid.height / h)
        mask &= rasterize([(grid.aoi, 1)], out_shape=(h, w), transform=transform, dtype="uint8").astype(bool)
    result = {}
    for name, arr in layers.items():
        vals = arr[mask & np.isfinite(arr)]
        if vals.size:
            low, high = np.percentile(vals, [2, 98])
            if high <= low:
                high = low + max(abs(low) * 0.01, 1e-6)
            result[name] = (float(low), float(high))
        else:
            result[name] = (0.0, 1.0)
    return result


def normalize(arr, limits):
    low, high = limits
    return np.clip((arr - low) / (high - low), 0, 1).astype("float32")


def odd_kernel(diameter_m: float, resolution_m: float) -> int:
    size = max(3, int(math.ceil(diameter_m / resolution_m)))
    return size if size % 2 else size + 1


def ratio_index(a, b):
    denominator = a + b
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(denominator) > 1e-10)
    result = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a - b, denominator, out=result, where=valid)
    return result


def local_std(arr, size):
    finite = np.isfinite(arr)
    values = np.where(finite, arr, 0).astype("float64")
    fraction = uniform_filter(finite.astype("float64"), size=size, mode="constant")
    sums = uniform_filter(values, size=size, mode="constant")
    squares = uniform_filter(values * values, size=size, mode="constant")
    mean = np.divide(sums, fraction, out=np.zeros_like(sums), where=fraction > 0)
    variance = np.divide(squares, fraction, out=np.zeros_like(sums), where=fraction > 0) - mean ** 2
    output = np.sqrt(np.maximum(variance, 0)).astype("float32")
    output[(fraction < 0.5) | ~finite] = np.nan
    return output


def make_features(layers, grid: Grid, normalizers):
    features = {k: v for k, v in layers.items() if k not in ("dsm", "dtm")}
    features["ndvi"] = ratio_index(layers["nir"], layers["red"])
    features["ndwi_green_nir"] = ratio_index(layers["green"], layers["nir"])
    features["ndre"] = ratio_index(layers["nir"], layers["red_edge"])
    features["ms_visible_brightness"] = (normalize(layers["green"], normalizers["green"]) + normalize(layers["red"], normalizers["red"])) / 2
    brightness = features["ms_visible_brightness"]
    if "rgb_red" in layers:
        rgb = np.stack([layers[n] for n in ("rgb_red", "rgb_green", "rgb_blue")])
        maximum, minimum = np.max(rgb, axis=0), np.min(rgb, axis=0)
        features["rgb_brightness"] = rgb.mean(axis=0)
        features["rgb_saturation"] = np.divide(maximum - minimum, maximum + 1e-10)
        brightness = features["rgb_brightness"]
    if "dsm" in layers:
        # Negative differences are retained so datum/alignment problems remain visible.
        features["height_above_ground_m"] = layers["dsm"] - layers["dtm"]
    texture_inputs = {"brightness": brightness, "nir": normalize(layers["nir"], normalizers["nir"]), "ndvi": features["ndvi"]}
    used_sizes = set()
    for diameter in TEXTURE_WINDOW_DIAMETERS_M:
        size = odd_kernel(diameter, grid.resolution_m)
        if size in used_sizes:
            continue
        used_sizes.add(size)
        for name, arr in texture_inputs.items():
            features[f"texture_{name}_std_{size}px"] = local_std(arr, size)
    if "dem" in layers:
        dem = layers["dem"]
        if min(dem.shape) >= 2:
            dy, dx = np.gradient(dem, grid.resolution_m)
            features["terrain_slope_degrees"] = np.degrees(np.arctan(np.hypot(dx, dy))).astype("float32")
        else:
            features["terrain_slope_degrees"] = np.full_like(dem, np.nan)
        size = odd_kernel(TERRAIN_RELIEF_WINDOW_M, grid.resolution_m)
        finite = np.isfinite(dem)
        high = maximum_filter(np.where(finite, dem, -np.inf), size=size, mode="constant", cval=-np.inf)
        low = minimum_filter(np.where(finite, dem, np.inf), size=size, mode="constant", cval=np.inf)
        with np.errstate(invalid="ignore"):
            relief = high - low
        fraction = uniform_filter(finite.astype("float32"), size=size, mode="constant")
        relief[(fraction < 0.5) | ~finite] = np.nan
        features["terrain_local_relief_m"] = relief
    return features


def segment_image(layers, valid, normalizers, resolution_m):
    if not valid.any():
        return np.zeros(valid.shape, dtype=np.int32)
    channels = [normalize(layers[name], normalizers[name])
                for name in ("green", "red", "red_edge", "nir")]
    if "rgb_red" in layers:
        channels += [normalize(layers[name], normalizers[name]) for name in ("rgb_red", "rgb_green", "rgb_blue")]
    image = np.stack(channels, axis=-1)
    image = np.nan_to_num(image, nan=0.5, posinf=1.0, neginf=0.0)
    n_segments = max(1, min(int(valid.sum() * resolution_m ** 2 / SEGMENT_SPACING_M ** 2), max(1, int(valid.sum() // 4))))
    result = slic(image, n_segments=n_segments, compactness=SLIC_COMPACTNESS,
                  sigma=SLIC_SIGMA_PIXELS, max_num_iter=SLIC_ITERATIONS,
                  convert2lab=False, enforce_connectivity=True,
                  min_size_factor=0.2, max_size_factor=3.0,
                  start_label=1, mask=None if valid.all() else valid, channel_axis=-1)
    result[~valid] = 0
    return result.astype(np.int32)


def aggregate_objects(labels, features, window: Window, grid: Grid):
    n = int(labels.max())
    ids = np.arange(1, n + 1)
    flat_labels = labels.ravel()
    pixels = np.bincount(flat_labels, minlength=n + 1)[1:].astype(np.int64)
    columns, names = [], []
    for name, arr in features.items():
        vals = arr.ravel().astype("float64")
        valid = np.isfinite(vals) & (flat_labels > 0)
        counts = np.bincount(flat_labels[valid], minlength=n + 1)[1:]
        sums = np.bincount(flat_labels[valid], weights=vals[valid], minlength=n + 1)[1:]
        squared = np.bincount(flat_labels[valid], weights=vals[valid] ** 2, minlength=n + 1)[1:]
        means = np.divide(sums, counts, out=np.full(n, np.nan), where=counts > 0)
        variance = np.divide(squared, counts, out=np.full(n, np.nan), where=counts > 0) - means ** 2
        columns.extend([means, np.sqrt(np.maximum(variance, 0))])
        names.extend([f"{name}_mean", f"{name}_std"])
        if name in ("dem", "rem", "height_above_ground_m"):
            columns.append(counts / pixels)
            names.append(f"{name}_valid_fraction")
    properties = regionprops_table(labels, properties=(
        "label", "centroid", "perimeter", "axis_major_length", "axis_minor_length",
        "eccentricity", "solidity", "extent"))
    if not np.array_equal(properties["label"], ids):
        raise RuntimeError("Object labels are not contiguous.")
    area = pixels * grid.resolution_m ** 2
    perimeter = np.maximum(properties["perimeter"] * grid.resolution_m, 4 * grid.resolution_m)
    major = np.maximum(properties["axis_major_length"] * grid.resolution_m, grid.resolution_m)
    minor = np.maximum(properties["axis_minor_length"] * grid.resolution_m, grid.resolution_m)
    columns.extend([area, perimeter, major, minor, np.minimum(major / minor, 100),
                    4 * np.pi * area / perimeter ** 2, properties["eccentricity"],
                    properties["solidity"], properties["extent"]])
    names.extend(["area_m2", "perimeter_m", "major_axis_m", "minor_axis_m", "elongation",
                  "shape_compactness", "eccentricity", "solidity", "extent"])
    transform = rasterio.windows.transform(window, grid.transform)
    x, y = transform * (properties["centroid-1"] + 0.5, properties["centroid-0"] + 0.5)
    return np.column_stack(columns).astype("float32"), names, pixels, np.asarray(x), np.asarray(y)


def rasterize_references(gdf, window, grid: Grid):
    transform = rasterio.windows.transform(window, grid.transform)
    h, w = int(window.height), int(window.width)
    footprint = box(*rasterio.windows.bounds(window, grid.transform))
    matches = gdf.sindex.query(footprint, predicate="intersects")
    if not len(matches):
        return np.zeros((h, w), dtype=np.int32)
    records = [(gdf.geometry.iloc[i], int(gdf.reference_id.iloc[i])) for i in matches]
    return rasterize(records, out_shape=(h, w), transform=transform, fill=0, dtype="int32")


def assign_object_references(labels, refs, gdf, exclusion, window, grid: Grid, pixels):
    n = len(pixels)
    target, role, group = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.int32)
    coverage, purity = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    reference_id = np.zeros(n, dtype=np.int32)
    mask = (labels > 0) & (refs > 0)
    if not mask.any():
        return target, role, group, coverage, purity, reference_id
    base = len(gdf) + 1
    keys, counts = np.unique(labels[mask].astype(np.int64) * base + refs[mask], return_counts=True)
    objects, polygons = keys // base, keys % base
    class_lookup = np.r_[0, gdf[CLASS_FIELD].to_numpy()]
    role_lookup = np.r_[0, gdf.role_code.to_numpy()]
    group_lookup = np.r_[0, gdf.group_code.to_numpy()]
    class_span = max(CLASS_NAMES) + 1
    class_counts = np.bincount(objects * class_span + class_lookup[polygons], weights=counts,
                               minlength=(n + 1) * class_span).reshape(n + 1, class_span)
    totals = np.bincount(objects, weights=counts, minlength=n + 1)[1:]
    dominant = np.argmax(class_counts[1:], axis=1)
    maximum = class_counts[1:][np.arange(n), dominant]
    coverage = (totals / pixels).astype(np.float32)
    purity = np.divide(maximum, totals, out=np.zeros(n), where=totals > 0).astype(np.float32)
    role_counts = np.bincount(objects * 4 + role_lookup[polygons], weights=counts,
                             minlength=(n + 1) * 4).reshape(n + 1, 4)[1:]
    gmin, gmax = np.full(n + 1, np.iinfo(np.int32).max), np.zeros(n + 1, dtype=np.int32)
    np.minimum.at(gmin, objects, group_lookup[polygons])
    np.maximum.at(gmax, objects, group_lookup[polygons])
    order = np.lexsort((polygons, -counts, objects))
    ordered_objects = objects[order]
    first = np.r_[True, np.diff(ordered_objects) != 0]
    reference_id[ordered_objects[first] - 1] = polygons[order][first]
    candidate = (coverage >= MIN_POLYGON_COVERAGE) & (purity >= MIN_LABEL_PURITY) & (gmin[1:] == gmax[1:])
    train = candidate & (role_counts[:, 1] > 0) & (role_counts[:, 2] == 0) & (role_counts[:, 3] == 0)
    validate = candidate & (role_counts[:, 2] > 0) & (role_counts[:, 1] == 0) & (role_counts[:, 3] == 0)
    if train.any():
        transform = rasterio.windows.transform(window, grid.transform)
        excluded = rasterize([(exclusion, 1)], out_shape=labels.shape, transform=transform, dtype="uint8", all_touched=True)
        touching = np.bincount(labels[excluded > 0], minlength=n + 1)[1:] > 0
        train &= ~touching
    target[train | validate] = dominant[train | validate]
    role[train], role[validate] = 1, 2
    group[train | validate] = gmax[1:][train | validate]
    return target, role, group, coverage, purity, reference_id


def raster_profile(grid: Grid, dtype, count=1, nodata=0):
    return {"driver": "GTiff", "height": grid.height, "width": grid.width,
            "count": count, "dtype": dtype, "crs": grid.crs, "transform": grid.transform,
            "nodata": nodata, "compress": "deflate", "predictor": 3 if dtype == "float32" else 2,
            "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}


def windows(grid: Grid):
    for row in range(0, grid.height, TILE_SIZE_PIXELS):
        for col in range(0, grid.width, TILE_SIZE_PIXELS):
            yield Window(col, row, min(TILE_SIZE_PIXELS, grid.width - col), min(TILE_SIZE_PIXELS, grid.height - row))


def extract_objects(readers, grid: Grid, gdf, exclusion, normalizers, cache_dir, output):
    total = math.ceil(grid.width / TILE_SIZE_PIXELS) * math.ceil(grid.height / TILE_SIZE_PIXELS)
    largest_kernel = max([odd_kernel(d, grid.resolution_m) for d in TEXTURE_WINDOW_DIAMETERS_M] + [odd_kernel(TERRAIN_RELIEF_WINDOW_M, grid.resolution_m)])
    halo = max(CONTEXT_HALO_PIXELS, largest_kernel // 2 + 2)
    cache_files, training, feature_names = [], [], None
    offset = 0
    start = time.monotonic()
    with rasterio.open(output / "segments.tif", "w", **raster_profile(grid, "uint32")) as segment_file:
        for tile, window in enumerate(windows(grid), start=1):
            r, c, h, w = map(int, (window.row_off, window.col_off, window.height, window.width))
            r0, c0 = max(0, r - halo), max(0, c - halo)
            r1, c1 = min(grid.height, r + h + halo), min(grid.width, c + w + halo)
            padded = Window(c0, r0, c1 - c0, r1 - r0)
            core = (slice(r - r0, r - r0 + h), slice(c - c0, c - c0 + w))
            layers = read_layers(readers, padded)
            valid = np.logical_and.reduce([np.isfinite(layers[name]) for name in ("green", "red", "red_edge", "nir")])
            if grid.aoi is not None:
                valid &= rasterize([(grid.aoi, 1)], out_shape=valid.shape,
                                   transform=rasterio.windows.transform(padded, grid.transform), dtype="uint8").astype(bool)
            if not valid[core].any():
                segment_file.write(np.zeros((h, w), dtype=np.uint32), 1, window=window)
                log(f"Features {tile}/{total}: no valid imagery")
                continue
            padded_labels = segment_image(layers, valid, normalizers, grid.resolution_m)
            labels = connected_labels(padded_labels[core], background=0, connectivity=1).astype(np.int32)
            features = {name: arr[core] for name, arr in make_features(layers, grid, normalizers).items()}
            x, names, pixels, cx, cy = aggregate_objects(labels, features, window, grid)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("Feature order changed between tiles.")
            refs = rasterize_references(gdf, window, grid)
            target, role, group, coverage, purity, reference_id = assign_object_references(labels, refs, gdf, exclusion, window, grid, pixels)
            n = len(x)
            if offset + n >= np.iinfo(np.uint32).max:
                raise ValueError("Too many objects for uint32 labels. Use a smaller AOI or coarser segmentation.")
            object_ids = np.arange(offset + 1, offset + n + 1, dtype=np.uint32)
            lookup = np.r_[np.uint32(0), object_ids]
            segment_file.write(lookup[labels], 1, window=window)
            cache_path = cache_dir / f"tile_{tile:06d}.npz"
            np.savez_compressed(cache_path, x=x, ids=object_ids, pixels=pixels, cx=cx, cy=cy,
                                window=np.array([c, r, w, h]), target=target, role=role, group=group,
                                coverage=coverage, purity=purity, reference_id=reference_id)
            cache_files.append(cache_path)
            sampled = target > 0
            if sampled.any():
                training.append({"x": x[sampled], "y": target[sampled], "role": role[sampled],
                                 "group": group[sampled], "ids": object_ids[sampled],
                                 "pixels": pixels[sampled], "reference_id": reference_id[sampled]})
            offset += n
            log(f"Features {tile}/{total}: {n:,} objects; {sampled.sum():,} reference objects; {(time.monotonic() - start) / 60:.1f} min")
    if not cache_files:
        raise ValueError("No valid four-band multispectral pixels within the analysis area.")
    if not training:
        raise ValueError("No usable reference objects. Check polygon overlap, coverage/purity thresholds, segmentation, and validation buffer.")
    samples = {key: np.concatenate([part[key] for part in training]) for key in training[0]}
    return cache_files, feature_names, samples, offset


def fit_classifier(samples, feature_names, output, reference_classes):
    train = samples["role"] == 1
    validate = samples["role"] == 2
    classes = np.array(sorted(reference_classes), dtype=np.int32)
    counts = {int(c): int(np.sum(train & (samples["y"] == c))) for c in classes}
    insufficient = {c: n for c, n in counts.items() if n < MIN_TRAINING_OBJECTS_PER_CLASS}
    if insufficient:
        raise ValueError(f"Too few usable training objects per class: {insufficient}. Add polygons farther from the validation buffer, enlarge polygons, or inspect segmentation.")
    if not validate.any():
        raise ValueError("No usable validation objects. Enlarge validation polygons or inspect coverage/purity thresholds.")
    missing_validation = set(classes) - set(samples["y"][validate])
    if missing_validation and REQUIRE_ALL_CLASSES_IN_VALIDATION:
        raise ValueError(f"No usable validation objects for classes {sorted(missing_validation)}. Add validation examples or relax REQUIRE_ALL_CLASSES_IN_VALIDATION.")
    rng = np.random.default_rng(RANDOM_SEED)
    selected = []
    for class_id in classes:
        positions = np.flatnonzero(train & (samples["y"] == class_id))
        if MAX_TRAINING_OBJECTS_PER_CLASS and len(positions) > MAX_TRAINING_OBJECTS_PER_CLASS:
            positions = rng.choice(positions, MAX_TRAINING_OBJECTS_PER_CLASS, replace=False)
        selected.extend(positions.tolist())
    selected = np.asarray(selected, dtype=np.int64)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("rf", RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                      max_features=RF_MAX_FEATURES, class_weight="balanced_subsample",
                                      n_jobs=RF_N_JOBS, random_state=RANDOM_SEED)),
    ])
    log(f"Fitting Random Forest: {len(selected):,} training objects; {validate.sum():,} held-out objects; {len(feature_names)} features")
    model.fit(samples["x"][selected], samples["y"][selected])
    probabilities = model.predict_proba(samples["x"][validate])
    raw, accepted, score, margin = scored_predictions(probabilities, model.classes_)
    obj_classes = np.r_[model.classes_, UNCERTAIN_CLASS_ID]
    raw_cm = confusion_matrix(samples["y"][validate], raw, labels=obj_classes)
    accepted_cm = confusion_matrix(samples["y"][validate], accepted, labels=obj_classes)
    raw_metrics = save_metrics(raw_cm, obj_classes, "validation_object_raw", output)
    accepted_metrics = save_metrics(accepted_cm, obj_classes, "validation_object_accepted", output)
    frame = pd.DataFrame({"object_id": samples["ids"], "reference_class": samples["y"],
                          "split": np.where(train, "train", "validate"), "group_code": samples["group"],
                          "reference_id": samples["reference_id"], "area_m2": samples["pixels"]})
    frame.rename(columns={"area_m2": "pixel_count"}, inplace=True)
    frame["used_for_fit"] = False
    frame.loc[selected, "used_for_fit"] = True
    frame.to_csv(output / "reference_objects.csv", index=False)
    pd.DataFrame({"object_id": samples["ids"][validate], "reference_class": samples["y"][validate],
                  "predicted_class": raw, "accepted_class": accepted, "score": score,
                  "score_margin": margin}).to_csv(output / "validation_object_predictions.csv", index=False)
    importance_names = model.named_steps["imputer"].get_feature_names_out(feature_names)
    pd.DataFrame({"feature": importance_names, "impurity_importance": model.named_steps["rf"].feature_importances_}) \
        .sort_values("impurity_importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    log(f"Held-out object accuracy: {raw_metrics['overall_accuracy']:.3f}; accepted-map accuracy including uncertain: {accepted_metrics['overall_accuracy']:.3f}")
    return model, {"training_counts_before_cap": counts, "training_count_used": len(selected),
                   "validation_object_count": int(validate.sum()), "object_raw": raw_metrics,
                   "object_accepted": accepted_metrics}


def scored_predictions(probabilities, classes):
    indices = probabilities.argmax(axis=1)
    predicted = classes[indices].astype(np.uint8)
    largest = probabilities[np.arange(len(indices)), indices]
    second = np.partition(probabilities, -2, axis=1)[:, -2] if probabilities.shape[1] > 1 else np.zeros(len(indices))
    margin = largest - second
    accepted = predicted.copy()
    accepted[(largest < MIN_CLASS_SCORE) | (margin < MIN_SCORE_MARGIN)] = UNCERTAIN_CLASS_ID
    return predicted, accepted, largest.astype(np.float32), margin.astype(np.float32)


def save_metrics(cm, classes, prefix, output):
    names = [f"{c}: {CLASS_NAMES.get(int(c), 'Uncertain')}" for c in classes]
    pd.DataFrame(cm, index=pd.Index(names, name="reference_class"), columns=names).to_csv(output / f"{prefix}_confusion.csv")
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    correct = np.diag(cm)
    precision = np.divide(correct, predicted, out=np.full(len(classes), np.nan), where=predicted > 0)
    recall = np.divide(correct, support, out=np.full(len(classes), np.nan), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.full(len(classes), np.nan), where=(support + predicted) > 0)
    rows = pd.DataFrame({"class_id": classes, "class_name": [CLASS_NAMES.get(int(c), "Uncertain") for c in classes],
                         "precision": precision, "recall": recall, "f1": f1, "reference_count": support,
                         "predicted_count": predicted})
    rows.to_csv(output / f"{prefix}_metrics.csv", index=False)
    total = int(cm.sum())
    return {"overall_accuracy": float(correct.sum() / total) if total else None,
            "reference_count": total, "uncertain_fraction": float(cm[:, classes == UNCERTAIN_CLASS_ID].sum() / total) if total else None}


def update_pixel_confusion(cm, true, predicted, classes):
    lookup = np.full(256, -1, dtype=np.int32)
    lookup[classes] = np.arange(len(classes))
    rows, cols = lookup[true], lookup[predicted]
    if np.any(rows < 0) or np.any(cols < 0):
        raise RuntimeError("Validation class is missing from the classifier class list.")
    cm += np.bincount(rows * len(classes) + cols, minlength=len(classes) ** 2).reshape(cm.shape)


def write_predictions(model, cache_files, grid: Grid, gdf, output):
    classes = model.classes_.astype(np.int32)
    metric_classes = np.r_[classes, UNCERTAIN_CLASS_ID]
    raw_cm = np.zeros((len(metric_classes), len(metric_classes)), dtype=np.int64)
    accepted_cm = raw_cm.copy()
    accepted_counts, raw_counts = np.zeros(256, dtype=np.int64), np.zeros(256, dtype=np.int64)
    class_lookup = np.r_[0, gdf[CLASS_FIELD].to_numpy()]
    role_lookup = np.r_[0, gdf.role_code.to_numpy()]
    polygons_written = False
    start = time.monotonic()
    with ExitStack() as stack:
        # Raster files are initialized to their declared nodata values by GDAL.
        labels_file = stack.enter_context(rasterio.open(output / "segments.tif"))
        accepted_file = stack.enter_context(rasterio.open(output / "classification.tif", "w", **raster_profile(grid, "uint8")))
        raw_file = stack.enter_context(rasterio.open(output / "predicted_classes.tif", "w", **raster_profile(grid, "uint8")))
        confidence_file = stack.enter_context(rasterio.open(output / "confidence.tif", "w", **raster_profile(grid, "float32", nodata=-9999)))
        margin_file = stack.enter_context(rasterio.open(output / "score_margin.tif", "w", **raster_profile(grid, "float32", nodata=-9999)))
        water_file = stack.enter_context(rasterio.open(output / "visible_water.tif", "w", **raster_profile(grid, "uint8", nodata=255)))
        score_file = None
        if WRITE_CLASS_SCORE_RASTER:
            score_file = stack.enter_context(rasterio.open(output / "class_scores.tif", "w", **raster_profile(grid, "float32", count=len(classes), nodata=-9999)))
            for band, class_id in enumerate(classes, start=1):
                score_file.set_band_description(band, f"{class_id}: {CLASS_NAMES[int(class_id)]}")
            score_file.update_tags(score_interpretation="Uncalibrated Random Forest class scores")
        for file in (accepted_file, raw_file):
            file.write_colormap(1, CLASS_COLORS)
            file.update_tags(class_names=json.dumps(CLASS_NAMES), uncertain_class_id=UNCERTAIN_CLASS_ID)
        confidence_file.update_tags(score_interpretation="Largest uncalibrated Random Forest class score")
        water_file.update_tags(values="0=other visible cover, 1=water or whitewater, 255=unknown/no data")
        for tile, cache_path in enumerate(cache_files, start=1):
            with np.load(cache_path) as cached:
                data = {name: cached[name] for name in cached.files}
            window = Window(*data["window"].tolist())
            global_labels = labels_file.read(1, window=window)
            local_labels = np.where(global_labels > 0, global_labels.astype(np.int64) - int(data["ids"][0]) + 1, 0).astype(np.int32)
            probabilities = model.predict_proba(data["x"])
            predicted, accepted, score, margin = scored_predictions(probabilities, classes)
            predicted_grid, accepted_grid = np.r_[np.uint8(0), predicted][local_labels], np.r_[np.uint8(0), accepted][local_labels]
            raw_file.write(predicted_grid, 1, window=window)
            accepted_file.write(accepted_grid, 1, window=window)
            confidence_file.write(np.r_[np.float32(-9999), score][local_labels], 1, window=window)
            margin_file.write(np.r_[np.float32(-9999), margin][local_labels], 1, window=window)
            water = np.isin(accepted_grid, WATER_CLASS_IDS).astype(np.uint8)
            water[np.isin(accepted_grid, (0, UNCERTAIN_CLASS_ID, *OBSCURED_CLASS_IDS))] = 255
            water_file.write(water, 1, window=window)
            if score_file is not None:
                for band in range(len(classes)):
                    score_file.write(np.r_[np.float32(-9999), probabilities[:, band].astype(np.float32)][local_labels], band + 1, window=window)
            valid = local_labels > 0
            raw_counts += np.bincount(predicted_grid[valid], minlength=256)
            accepted_counts += np.bincount(accepted_grid[valid], minlength=256)
            refs = rasterize_references(gdf, window, grid)
            validation = valid & (role_lookup[refs] == 2)
            if validation.any():
                true = class_lookup[refs[validation]]
                update_pixel_confusion(raw_cm, true, predicted_grid[validation], metric_classes)
                update_pixel_confusion(accepted_cm, true, accepted_grid[validation], metric_classes)
            frame = pd.DataFrame({"object_id": data["ids"].astype(np.int64), "predicted_class": predicted,
                                  "accepted_class": accepted, "class_name": [CLASS_NAMES.get(int(c), "Uncertain") for c in accepted],
                                  "score": score, "score_margin": margin,
                                  "area_m2": data["pixels"] * grid.resolution_m ** 2,
                                  "centroid_x": data["cx"], "centroid_y": data["cy"],
                                  "reference_class": data["target"], "reference_role": data["role"],
                                  "reference_id": data["reference_id"], "reference_coverage": data["coverage"],
                                  "reference_purity": data["purity"]})
            frame.to_csv(output / "object_predictions.csv", mode="w" if tile == 1 else "a", header=tile == 1, index=False)
            if EXPORT_OBJECT_POLYGONS:
                geometries, local_ids = [], []
                transform = rasterio.windows.transform(window, grid.transform)
                for geom, value in shapes(local_labels, mask=valid, connectivity=4, transform=transform):
                    local_ids.append(int(value) - 1)
                    geometries.append(shape(geom))
                objects = gpd.GeoDataFrame(frame.iloc[local_ids].reset_index(drop=True), geometry=geometries, crs=grid.crs)
                if not objects.geometry.is_valid.all():
                    raise RuntimeError("Invalid polygon geometry in exported segmentation.")
                objects.to_file(output / "classified_objects.gpkg", layer="objects", driver="GPKG", engine="pyogrio", mode="a" if polygons_written else "w")
                polygons_written = True
            log(f"Prediction {tile}/{len(cache_files)}: {len(frame):,} objects; {(time.monotonic() - start) / 60:.1f} min")
    raw_metrics = save_metrics(raw_cm, metric_classes, "validation_pixel_raw", output)
    accepted_metrics = save_metrics(accepted_cm, metric_classes, "validation_pixel_accepted", output)
    if not raw_metrics["reference_count"]:
        raise ValueError("No valid imagery pixels in validation polygons.")
    total = int(accepted_counts.sum())
    area_rows = []
    for class_id in sorted(set(classes.tolist()) | {UNCERTAIN_CLASS_ID}):
        area_rows.append({"class_id": class_id, "class_name": CLASS_NAMES.get(class_id, "Uncertain"),
                          "accepted_pixels": int(accepted_counts[class_id]),
                          "accepted_area_m2": accepted_counts[class_id] * grid.resolution_m ** 2,
                          "accepted_area_acres": accepted_counts[class_id] * grid.resolution_m ** 2 / 4046.8564224,
                          "percent_valid_imagery": 100 * accepted_counts[class_id] / total,
                          "raw_predicted_area_m2": raw_counts[class_id] * grid.resolution_m ** 2})
    pd.DataFrame(area_rows).to_csv(output / "mapped_area_summary.csv", index=False)
    log(f"Held-out reference-pixel accuracy: {raw_metrics['overall_accuracy']:.3f}; uncertain map area: {100 * accepted_counts[UNCERTAIN_CLASS_ID] / total:.1f}%")
    return {"pixel_raw": raw_metrics, "pixel_accepted": accepted_metrics,
            "valid_pixel_count": total, "mapped_area_m2": total * grid.resolution_m ** 2,
            "uncertain_area_fraction": float(accepted_counts[UNCERTAIN_CLASS_ID] / total)}


def settings_snapshot():
    names = [name for name in globals() if name.isupper() and not name.startswith("_")]
    return {name: globals()[name] for name in names}


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> Path | None:
    start = time.monotonic()
    check_settings()
    specs = source_specs()
    grid = make_grid(specs["green"][0])
    log(f"Grid: {grid.width:,} x {grid.height:,}; resolution {grid.resolution_m:.4f} m; {grid.crs}")
    if not Path(TRAINING_PATH).exists() and CREATE_TRAINING_TEMPLATE_IF_MISSING:
        if Path(TRAINING_PATH).suffix.lower() != ".gpkg":
            raise ValueError("Automatic training templates require a .gpkg path.")
        create_training_template(Path(TRAINING_PATH), grid)
        return None
    references, exclusion, split_method = load_references(grid)
    log(f"Validation: {split_method}; exclusion buffer {VALIDATION_BUFFER_M:g} m")
    if SEGMENT_SPACING_M < grid.resolution_m * 2:
        log("NOTE: Segment spacing is below two pixels; actual objects will be larger.")
    output = Path(OUTPUT_ROOT) / datetime.now().strftime("classification_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    references.to_file(output / "training_polygons_used.gpkg", layer="references", driver="GPKG", engine="pyogrio")
    versions = {}
    for package in ("numpy", "pandas", "rasterio", "geopandas", "shapely", "scipy", "scikit-image", "scikit-learn", "pyogrio"):
        versions[package] = importlib.metadata.version(package)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings_snapshot(),
              "versions": versions, "crs": grid.crs.to_wkt(), "resolution_m": grid.resolution_m,
              "width": grid.width, "height": grid.height, "validation_split_method": split_method,
              "limitations": ["Scores are uncalibrated, not accuracy probabilities.",
                              "Objects stop at tile boundaries; halos do not eliminate seams.",
                              "Surface cover includes visible wood only; no volume or piece counts.",
                              "Whitewater identifies visible foam, not a hydraulic habitat class.",
                              "Mapped areas are not statistically corrected for classification error.",
                              "Validation represents the labeled polygons and their conditions."]}
    (output / "run_report.json").write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    try:
        with TemporaryDirectory(prefix="feature_cache_", dir=output) as temporary:
            with ExitStack() as stack:
                readers = open_aligned_sources(specs, grid, stack)
                normalizers = find_normalizers(readers, grid)
                cache_files, feature_names, samples, object_count = extract_objects(
                    readers, grid, references, exclusion, normalizers, Path(temporary), output)
            active_classes = set(references.loc[references.role_code != 3, CLASS_FIELD])
            model, validation = fit_classifier(samples, feature_names, output, active_classes)
            report.update({"feature_names": feature_names, "object_count": object_count,
                           "normalization_percentiles": normalizers, "validation": validation})
            joblib.dump({"pipeline": model, "feature_names": feature_names,
                         "class_names": CLASS_NAMES, "trained_class_ids": model.classes_,
                         "normalizers": normalizers, "settings": settings_snapshot(),
                         "training_scope": "training objects only; validation excluded",
                         "versions": versions}, output / "trained_model.joblib", compress=3)
            mapping = write_predictions(model, cache_files, grid, references, output)
            report["validation"].update({k: v for k, v in mapping.items() if k.startswith("pixel_")})
            report["mapping"] = {k: v for k, v in mapping.items() if not k.startswith("pixel_")}
        report["elapsed_minutes"] = (time.monotonic() - start) / 60
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        (output / "run_report.json").write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    log(f"Complete. Outputs: {output}")
    return output


if __name__ == "__main__":
    main()
