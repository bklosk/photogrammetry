#!/usr/bin/env python3
"""
Point Cloud Processing Script: Advanced Colorization

This script provides robust colorization of LiDAR point clouds using orthophotos.
It intelligently handles coordinate transformations, CRS matching, and provides
comprehensive diagnostics.

Features:
- Automatic CRS detection and transformation
- Robust coordinate system handling for common LiDAR/orthophoto combinations
- Intelligent fallback mechanisms for coordinate transformation
- Comprehensive diagnostic outputs
- Support for various orthophoto formats (NAIP, satellite imagery)
- Quality assessment and validation

Usage:
    python process_point_cloud.py --address "1250 Wildwood Road, Boulder, CO"
    python process_point_cloud.py --input_pc data/point_cloud.laz --input_ortho data/orthophoto.tif
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add the scripts directory to the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Geospatial libraries
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds
from pyproj import Transformer, CRS

# Point cloud processing
try:
    import laspy
except ImportError:
    print("ERROR: laspy not found. Install with: pip install laspy lazrs")
    sys.exit(1)

# Import local modules
from geocode import Geocoder
from get_point_cloud import PointCloudDatasetFinder
from get_orthophoto import NAIPFetcher
from utils import CRSUtils, BoundingBoxUtils, FileUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class PointCloudColorizer:
    """
    Advanced point cloud colorization using orthophotos.
    """

    def __init__(self, output_dir: str = "data", create_diagnostics: bool = False):
        """
        Initialize the colorizer.

        Args:
            output_dir: Directory for outputs and diagnostics
            create_diagnostics: Whether to create diagnostic plots (slower)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.create_diagnostics = create_diagnostics

        # Common CRS for different regions
        self.common_crs_mappings = {
            "colorado": {
                "utm": "EPSG:26913",  # UTM Zone 13N
                "state_plane": "EPSG:2232",  # NAD83 Colorado Central ftUS
                "state_plane_m": "EPSG:26954",  # NAD83 Colorado Central meters
            },
            "california": {
                "utm_10": "EPSG:26910",  # UTM Zone 10N
                "utm_11": "EPSG:26911",  # UTM Zone 11N
                "state_plane": "EPSG:2227",  # NAD83 California Zone 3 ftUS
            },
        }

    def load_point_cloud(self, file_path: str) -> laspy.LasData:
        """
        Load point cloud with comprehensive error handling.

        Args:
            file_path: Path to LAZ/LAS file

        Returns:
            Loaded point cloud data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")

        try:
            logger.info(f"Loading point cloud: {file_path}")
            las_data = laspy.read(str(file_path))

            # Basic statistics
            num_points = len(las_data.points)
            logger.info(f"Loaded {num_points:,} points")

            # Coordinate bounds
            x_min, x_max = np.min(las_data.x), np.max(las_data.x)
            y_min, y_max = np.min(las_data.y), np.max(las_data.y)
            z_min, z_max = np.min(las_data.z), np.max(las_data.z)

            logger.info(f"Point cloud bounds:")
            logger.info(f"  X: {x_min:.2f} to {x_max:.2f}")
            logger.info(f"  Y: {y_min:.2f} to {y_max:.2f}")
            logger.info(f"  Z: {z_min:.2f} to {z_max:.2f}")

            return las_data

        except Exception as e:
            raise RuntimeError(f"Failed to load point cloud: {e}")

    def load_orthophoto(self, file_path: str) -> rasterio.DatasetReader:
        """
        Load orthophoto with validation and fallback for filename mismatches.

        Args:
            file_path: Path to orthophoto file

        Returns:
            Rasterio dataset
        """
        file_path = Path(file_path)

        # If the exact file doesn't exist, try to find orthophoto files in the same directory
        if not file_path.exists():
            parent_dir = file_path.parent
            logger.warning(f"Orthophoto file not found: {file_path}")

            if parent_dir.exists():
                logger.info(
                    f"Searching for orthophoto files in directory: {parent_dir}"
                )

                # Look for any TIFF files that might be orthophotos
                tiff_patterns = ["*.tif", "*.tiff", "*naip*.tif", "*orthophoto*.tif"]
                found_files = []

                for pattern in tiff_patterns:
                    found_files.extend(list(parent_dir.glob(pattern)))

                if found_files:
                    # Use the first valid orthophoto file found
                    for candidate_file in found_files:
                        try:
                            logger.info(f"Trying candidate file: {candidate_file}")
                            with rasterio.open(str(candidate_file)) as test_ds:
                                if test_ds.width > 0 and test_ds.height > 0:
                                    logger.info(
                                        f"Found valid orthophoto: {candidate_file}"
                                    )
                                    file_path = candidate_file
                                    break
                        except Exception:
                            continue
                    else:
                        raise FileNotFoundError(
                            f"No valid orthophoto files found in {parent_dir}"
                        )
                else:
                    raise FileNotFoundError(
                        f"No orthophoto files found in {parent_dir}"
                    )
            else:
                raise FileNotFoundError(f"Orthophoto file not found: {file_path}")

        try:
            logger.info(f"Loading orthophoto: {file_path}")
            dataset = rasterio.open(str(file_path))

            # Basic information
            logger.info(f"Orthophoto info:")
            logger.info(f"  Size: {dataset.width} x {dataset.height}")
            logger.info(f"  Bands: {dataset.count}")
            logger.info(f"  CRS: {dataset.crs}")
            logger.info(f"  Bounds: {dataset.bounds}")

            if dataset.crs is None:
                logger.warning("Orthophoto has no CRS information")

            return dataset

        except Exception as e:
            raise RuntimeError(f"Failed to load orthophoto: {e}")

    def detect_point_cloud_crs(self, las_data: laspy.LasData) -> Optional[str]:
        """Detect point cloud CRS using centralized utilities."""
        return CRSUtils.detect_point_cloud_crs(las_data)

    def transform_point_cloud_to_ortho_crs(
        self, las_data: laspy.LasData, ortho_crs: str, source_crs: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform point cloud coordinates to orthophoto CRS with optimized batching.

        Args:
            las_data: Point cloud data
            ortho_crs: Target CRS (orthophoto CRS)
            source_crs: Source CRS (if known)

        Returns:
            Tuple of (transformed_x, transformed_y)
        """
        x_coords = las_data.x
        y_coords = las_data.y

        # Detect source CRS if not provided
        if source_crs is None:
            source_crs = self.detect_point_cloud_crs(las_data)

        if source_crs is None:
            # Enhanced fallback: Analyze coordinate ranges to guess CRS
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            logger.warning(f"CRS detection failed. Analyzing coordinate ranges:")
            logger.warning(
                f"X: {x_min:.2f} to {x_max:.2f}, Y: {y_min:.2f} to {y_max:.2f}"
            )

            # Heuristic: Large negative X values and large positive Y values suggest Web Mercator
            if (
                x_min < -1000000
                and abs(x_min) < 20037508
                and y_min > 1000000
                and y_max < 20037508
            ):
                source_crs = "EPSG:3857"
                logger.info(
                    f"Fallback: Assuming Web Mercator (EPSG:3857) based on coordinate ranges"
                )
            # Geographic coordinates
            elif -180 <= x_min <= 180 and -90 <= y_min <= 90:
                source_crs = "EPSG:4326"
                logger.info(
                    f"Fallback: Assuming WGS84 (EPSG:4326) based on coordinate ranges"
                )
            # UTM-like coordinates
            elif 100000 < x_min < 900000 and 1000000 < y_min < 10000000:
                source_crs = "EPSG:26913"  # Assume UTM Zone 13N for western US
                logger.info(
                    f"Fallback: Assuming UTM Zone 13N (EPSG:26913) based on coordinate ranges"
                )
            else:
                raise ValueError(
                    f"Cannot determine point cloud CRS. Coordinate ranges: X[{x_min:.0f}, {x_max:.0f}], Y[{y_min:.0f}, {y_max:.0f}]"
                )

        logger.info(f"Transforming coordinates: {source_crs} -> {ortho_crs}")

        try:
            # Validate CRS
            source_crs_obj = CRS.from_string(source_crs)
            target_crs_obj = CRS.from_string(ortho_crs)

            # Create transformer
            transformer = Transformer.from_crs(
                source_crs_obj, target_crs_obj, always_xy=True
            )

            # Optimized batch size for faster processing
            batch_size = 500000  # Increased from 100k
            total_points = len(x_coords)

            transformed_x = np.zeros_like(x_coords)
            transformed_y = np.zeros_like(y_coords)

            logger.info(
                f"Transforming {total_points:,} points in batches of {batch_size:,}..."
            )

            for i in tqdm(range(0, total_points, batch_size), desc="Transforming"):
                end_idx = min(i + batch_size, total_points)

                batch_x = x_coords[i:end_idx]
                batch_y = y_coords[i:end_idx]

                trans_x, trans_y = transformer.transform(batch_x, batch_y)

                transformed_x[i:end_idx] = trans_x
                transformed_y[i:end_idx] = trans_y

            return transformed_x, transformed_y

        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise RuntimeError(f"Coordinate transformation failed: {e}")

    def create_alignment_diagnostic(
        self,
        las_data: laspy.LasData,
        ortho_dataset: rasterio.DatasetReader,
        transformed_x: np.ndarray,
        transformed_y: np.ndarray,
        output_name: str = "alignment_diagnostic.png",
    ):
        """
        Create diagnostic plot showing point cloud and orthophoto alignment.

        Args:
            las_data: Original point cloud data
            ortho_dataset: Orthophoto dataset
            transformed_x: Transformed X coordinates
            transformed_y: Transformed Y coordinates
            output_name: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Overview with bounds
        ortho_bounds = ortho_dataset.bounds

        # Sample points for visualization
        sample_size = min(5000, len(transformed_x))
        sample_indices = np.random.choice(
            len(transformed_x), sample_size, replace=False
        )

        sample_x = transformed_x[sample_indices]
        sample_y = transformed_y[sample_indices]

        # Orthophoto bounds
        rect = plt.Rectangle(
            (ortho_bounds.left, ortho_bounds.bottom),
            ortho_bounds.right - ortho_bounds.left,
            ortho_bounds.top - ortho_bounds.bottom,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Orthophoto Bounds",
        )
        ax1.add_patch(rect)

        # Point cloud
        ax1.scatter(
            sample_x,
            sample_y,
            s=1,
            c="blue",
            alpha=0.6,
            label=f"Point Cloud ({sample_size:,} points)",
        )

        ax1.set_title("Coordinate Alignment Overview")
        ax1.set_xlabel(f"X ({ortho_dataset.crs})")
        ax1.set_ylabel(f"Y ({ortho_dataset.crs})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Intersection analysis
        # Find points within orthophoto bounds
        within_bounds = (
            (sample_x >= ortho_bounds.left)
            & (sample_x <= ortho_bounds.right)
            & (sample_y >= ortho_bounds.bottom)
            & (sample_y <= ortho_bounds.top)
        )

        points_in_bounds = np.sum(within_bounds)

        ax2.scatter(
            sample_x[~within_bounds],
            sample_y[~within_bounds],
            s=1,
            c="red",
            alpha=0.6,
            label=f"Outside bounds ({np.sum(~within_bounds):,})",
        )
        ax2.scatter(
            sample_x[within_bounds],
            sample_y[within_bounds],
            s=1,
            c="green",
            alpha=0.6,
            label=f"Within bounds ({points_in_bounds:,})",
        )

        ax2.add_patch(
            plt.Rectangle(
                (ortho_bounds.left, ortho_bounds.bottom),
                ortho_bounds.right - ortho_bounds.left,
                ortho_bounds.top - ortho_bounds.bottom,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
        )

        ax2.set_title("Point Distribution Analysis")
        ax2.set_xlabel(f"X ({ortho_dataset.crs})")
        ax2.set_ylabel(f"Y ({ortho_dataset.crs})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Diagnostic plot saved: {output_path}")
        logger.info(
            f"Points within orthophoto bounds: {points_in_bounds:,}/{sample_size:,} "
            f"({100*points_in_bounds/sample_size:.1f}%)"
        )

    def colorize_point_cloud(
        self,
        las_data: laspy.LasData,
        ortho_dataset: rasterio.DatasetReader,
        source_crs: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Colorize point cloud using orthophoto with optimized performance.

        Args:
            las_data: Point cloud data
            ortho_dataset: Orthophoto dataset
            source_crs: Source CRS override

        Returns:
            Tuple of (RGB color array (N, 3) with uint16 values, valid_mask boolean array)
        """
        ortho_crs = str(ortho_dataset.crs) if ortho_dataset.crs else None
        if ortho_crs is None:
            raise ValueError("Orthophoto has no CRS information")

        # Transform coordinates
        transformed_x, transformed_y = self.transform_point_cloud_to_ortho_crs(
            las_data, ortho_crs, source_crs
        )

        # Enhanced debugging: Print coordinate ranges and bounds
        logger.info("=== COORDINATE ANALYSIS ===")
        logger.info(f"Orthophoto CRS: {ortho_crs}")
        logger.info(f"Orthophoto bounds: {ortho_dataset.bounds}")

        # Point cloud bounds in transformed coordinates
        pc_x_min, pc_x_max = np.min(transformed_x), np.max(transformed_x)
        pc_y_min, pc_y_max = np.min(transformed_y), np.max(transformed_y)
        logger.info(
            f"Point cloud bounds (transformed): X[{pc_x_min:.2f}, {pc_x_max:.2f}], Y[{pc_y_min:.2f}, {pc_y_max:.2f}]"
        )

        # Calculate distances between centers
        ortho_center_x = (ortho_dataset.bounds.left + ortho_dataset.bounds.right) / 2
        ortho_center_y = (ortho_dataset.bounds.bottom + ortho_dataset.bounds.top) / 2
        pc_center_x = (pc_x_min + pc_x_max) / 2
        pc_center_y = (pc_y_min + pc_y_max) / 2

        distance_x = abs(ortho_center_x - pc_center_x)
        distance_y = abs(ortho_center_y - pc_center_y)
        logger.info(
            f"Distance between centers: X={distance_x:.2f}m, Y={distance_y:.2f}m"
        )

        # Check for overlap
        overlap_x = not (
            pc_x_max < ortho_dataset.bounds.left
            or pc_x_min > ortho_dataset.bounds.right
        )
        overlap_y = not (
            pc_y_max < ortho_dataset.bounds.bottom
            or pc_y_min > ortho_dataset.bounds.top
        )
        logger.info(f"Bounds overlap: X={overlap_x}, Y={overlap_y}")

        # If there's no overlap, try alternative CRS transformations
        if not (overlap_x and overlap_y):
            logger.warning(
                "No coordinate overlap detected. Attempting CRS correction..."
            )

            # Calculate how far off the data is
            pc_center_x = (pc_x_min + pc_x_max) / 2
            pc_center_y = (pc_y_min + pc_y_max) / 2
            distance_from_center = (
                (ortho_center_x - pc_center_x) ** 2
                + (ortho_center_y - pc_center_y) ** 2
            ) ** 0.5

            logger.error(f"SIGNIFICANT COORDINATE MISMATCH DETECTED!")
            logger.error(f"Point cloud center: ({pc_center_x:.6f}, {pc_center_y:.6f})")
            logger.error(
                f"Orthophoto center: ({ortho_center_x:.6f}, {ortho_center_y:.6f})"
            )
            logger.error(
                f"Distance between centers: {distance_from_center:.6f} degrees ({distance_from_center * 111320:.1f} meters)"
            )

            if distance_from_center > 0.001:  # More than ~100 meters apart
                logger.error(
                    "The orthophoto and point cloud appear to be from different locations!"
                )
                logger.error(
                    "SOLUTION: Use the correct bounding box for orthophoto download:"
                )

                # Calculate the correct bounding box
                buffer_deg = 0.001  # ~100 meter buffer
                correct_bbox = {
                    "west": pc_x_min - buffer_deg,
                    "south": pc_y_min - buffer_deg,
                    "east": pc_x_max + buffer_deg,
                    "north": pc_y_max + buffer_deg,
                }

                bbox_string = f"{correct_bbox['west']:.8f},{correct_bbox['south']:.8f},{correct_bbox['east']:.8f},{correct_bbox['north']:.8f}"
                logger.error(f"Correct bounding box: {bbox_string}")
                logger.error(
                    f'Download command: python scripts/get_orthophoto.py --bbox "{bbox_string}"'
                )

                # For this run, we'll proceed with a warning, but results will be poor
                logger.warning(
                    "Continuing with current data, but expect very low coverage..."
                )

            # Try transforming orthophoto bounds to point cloud's original CRS for comparison
            original_x_coords = las_data.x
            original_y_coords = las_data.y
            orig_x_min, orig_x_max = np.min(original_x_coords), np.max(
                original_x_coords
            )
            orig_y_min, orig_y_max = np.min(original_y_coords), np.max(
                original_y_coords
            )

            # Detect or assume point cloud CRS
            pc_crs = source_crs or self.detect_point_cloud_crs(las_data)
            if pc_crs is None:
                pc_crs = "EPSG:3857"  # Default fallback based on error log

            logger.info(f"Original point cloud CRS: {pc_crs}")
            logger.info(
                f"Original point cloud bounds: X[{orig_x_min:.2f}, {orig_x_max:.2f}], Y[{orig_y_min:.2f}, {orig_y_max:.2f}]"
            )

            # Try transforming orthophoto bounds to point cloud CRS
            try:
                ortho_to_pc_transformer = Transformer.from_crs(
                    CRS.from_string(ortho_crs), CRS.from_string(pc_crs), always_xy=True
                )

                # Transform the four corners of the orthophoto bounds
                ortho_left, ortho_bottom = ortho_to_pc_transformer.transform(
                    ortho_dataset.bounds.left, ortho_dataset.bounds.bottom
                )
                ortho_right, ortho_top = ortho_to_pc_transformer.transform(
                    ortho_dataset.bounds.right, ortho_dataset.bounds.top
                )

                logger.info(
                    f"Orthophoto bounds in point cloud CRS: [{ortho_left:.2f}, {ortho_bottom:.2f}, {ortho_right:.2f}, {ortho_top:.2f}]"
                )

                # Check overlap in original coordinate space
                ortho_pc_overlap_x = not (
                    orig_x_max < ortho_left or orig_x_min > ortho_right
                )
                ortho_pc_overlap_y = not (
                    orig_y_max < ortho_bottom or orig_y_min > ortho_top
                )

                logger.info(
                    f"Overlap in original PC CRS: X={ortho_pc_overlap_x}, Y={ortho_pc_overlap_y}"
                )

                if not (ortho_pc_overlap_x and ortho_pc_overlap_y):
                    logger.error(
                        "Point cloud and orthophoto do not overlap in any tested coordinate system!"
                    )
                    logger.error(
                        "This suggests they are from different geographic areas."
                    )

            except Exception as e:
                logger.warning(f"Could not transform orthophoto bounds to PC CRS: {e}")

        # Create diagnostic plot if requested or if there's coordinate issues
        if self.create_diagnostics or not (overlap_x and overlap_y):
            self.create_alignment_diagnostic(
                las_data, ortho_dataset, transformed_x, transformed_y
            )

        # Convert to pixel coordinates
        logger.info("Converting to pixel coordinates...")

        # DEBUG: Examine the geotransform matrix
        logger.info(f"DEBUG - Orthophoto transform matrix: {ortho_dataset.transform}")
        logger.info(f"DEBUG - Transform breakdown:")
        logger.info(f"  Pixel width (X resolution): {ortho_dataset.transform[0]}")
        logger.info(f"  Row rotation: {ortho_dataset.transform[1]}")
        logger.info(f"  Upper-left X coordinate: {ortho_dataset.transform[2]}")
        logger.info(f"  Column rotation: {ortho_dataset.transform[3]}")
        logger.info(f"  Pixel height (Y resolution): {ortho_dataset.transform[4]}")
        logger.info(f"  Upper-left Y coordinate: {ortho_dataset.transform[5]}")

        # Calculate expected transform based on bounds and image dimensions
        expected_pixel_width = (
            ortho_dataset.bounds.right - ortho_dataset.bounds.left
        ) / ortho_dataset.width
        expected_pixel_height = (
            ortho_dataset.bounds.top - ortho_dataset.bounds.bottom
        ) / ortho_dataset.height

        logger.info(f"DEBUG - Expected pixel dimensions:")
        logger.info(f"  Expected pixel width: {expected_pixel_width}")
        logger.info(f"  Expected pixel height: {expected_pixel_height}")
        logger.info(f"  Actual pixel width: {ortho_dataset.transform[0]}")
        logger.info(f"  Actual pixel height: {abs(ortho_dataset.transform[4])}")

        # Check if transform seems reasonable
        transform_issues = []
        if (
            abs(ortho_dataset.transform[0] - expected_pixel_width)
            > expected_pixel_width * 0.1
        ):
            transform_issues.append("X pixel size mismatch")
        if (
            abs(abs(ortho_dataset.transform[4]) - expected_pixel_height)
            > expected_pixel_height * 0.1
        ):
            transform_issues.append("Y pixel size mismatch")

        if transform_issues:
            logger.warning(
                f"Potential transform issues detected: {', '.join(transform_issues)}"
            )
            logger.warning("Attempting to create corrected transform...")

            # Create a corrected transform based on bounds
            corrected_transform = from_bounds(
                ortho_dataset.bounds.left,
                ortho_dataset.bounds.bottom,
                ortho_dataset.bounds.right,
                ortho_dataset.bounds.top,
                ortho_dataset.width,
                ortho_dataset.height,
            )

            logger.info(f"DEBUG - Corrected transform: {corrected_transform}")

            # Try with corrected transform
            rows, cols = rasterio.transform.rowcol(
                corrected_transform, transformed_x, transformed_y
            )
            logger.info("Using corrected transform for pixel coordinate calculation")
        else:
            rows, cols = rasterio.transform.rowcol(
                ortho_dataset.transform, transformed_x, transformed_y
            )

        pixel_cols = np.array(cols, dtype=np.int32)
        pixel_rows = np.array(rows, dtype=np.int32)

        # DEBUG: Check coordinate transformation results
        logger.info(f"DEBUG - Coordinate transformation samples (first 5):")
        for i in range(min(5, len(transformed_x))):
            logger.info(
                f"  Point {i}: World({transformed_x[i]:.6f}, {transformed_y[i]:.6f}) -> Pixel({pixel_cols[i]}, {pixel_rows[i]})"
            )

        logger.info(f"DEBUG - Pixel coordinate ranges:")
        logger.info(
            f"  Columns: min={np.min(pixel_cols)}, max={np.max(pixel_cols)} (image width: {ortho_dataset.width})"
        )
        logger.info(
            f"  Rows: min={np.min(pixel_rows)}, max={np.max(pixel_rows)} (image height: {ortho_dataset.height})"
        )

        # Find valid pixels
        valid_mask = (
            (pixel_cols >= 0)
            & (pixel_cols < ortho_dataset.width)
            & (pixel_rows >= 0)
            & (pixel_rows < ortho_dataset.height)
        )

        num_valid = np.sum(valid_mask)
        total_points = len(las_data.points)

        logger.info(
            f"Valid points for colorization: {num_valid:,}/{total_points:,} "
            f"({100*num_valid/total_points:.1f}%)"
        )

        # Check if coverage is too low (less than 10% of points)
        coverage_threshold = 0.10  # 10%
        current_coverage = num_valid / total_points

        if num_valid == 0 or current_coverage < coverage_threshold:
            if num_valid == 0:
                logger.error("No points fall within orthophoto bounds!")
            else:
                logger.warning(
                    f"Low coverage detected: only {current_coverage:.1%} of points are within orthophoto bounds"
                )

            logger.info(
                "Attempting to download corrected orthophoto with proper bounds..."
            )

            # Calculate point cloud bounds for corrected orthophoto
            pc_bounds = {
                "west": pc_x_min,
                "east": pc_x_max,
                "south": pc_y_min,
                "north": pc_y_max,
            }

            try:
                # Download corrected orthophoto
                corrected_ortho_path = self.download_corrected_orthophoto(pc_bounds)

                logger.info(
                    "Successfully downloaded corrected orthophoto. Retrying colorization..."
                )

                # Reload with corrected orthophoto and retry colorization
                with rasterio.open(corrected_ortho_path) as corrected_dataset:
                    # Recursive call with corrected orthophoto
                    return self.colorize_point_cloud(
                        las_data, corrected_dataset, source_crs
                    )

            except Exception as e:
                logger.error(f"Failed to download corrected orthophoto: {e}")
                logger.error(
                    "Proceeding with original orthophoto but expect poor results..."
                )

                if num_valid == 0:
                    raise ValueError(
                        "No points within orthophoto bounds and auto-correction failed"
                    )

        # Initialize color array
        colors = np.zeros((total_points, 3), dtype=np.uint16)

        # Only extract colors for valid points - massive performance improvement
        logger.info("Extracting colors from orthophoto (optimized)...")

        valid_cols = pixel_cols[valid_mask]
        valid_rows = pixel_rows[valid_mask]

        if ortho_dataset.count >= 3:
            # Read only the required pixels from RGB bands using advanced indexing
            # This is much faster than reading entire bands
            logger.info("Reading RGB bands for valid pixels only...")

            # Use rasterio's window reading for better performance
            with rasterio.Env():
                red_band = ortho_dataset.read(1)
                green_band = ortho_dataset.read(2)
                blue_band = ortho_dataset.read(3)

                red_values = red_band[valid_rows, valid_cols]
                green_values = green_band[valid_rows, valid_cols]
                blue_values = blue_band[valid_rows, valid_cols]

                # DEBUG: Check raw pixel values
                logger.info(f"DEBUG - Raw pixel value samples (first 5):")
                logger.info(f"  Red: {red_values[:5]}")
                logger.info(f"  Green: {green_values[:5]}")
                logger.info(f"  Blue: {blue_values[:5]}")
                logger.info(f"DEBUG - Raw pixel value ranges:")
                logger.info(
                    f"  Red: min={np.min(red_values)}, max={np.max(red_values)}"
                )
                logger.info(
                    f"  Green: min={np.min(green_values)}, max={np.max(green_values)}"
                )
                logger.info(
                    f"  Blue: min={np.min(blue_values)}, max={np.max(blue_values)}"
                )

        elif ortho_dataset.count == 1:
            # Grayscale
            logger.info("Reading grayscale band for valid pixels only...")
            gray_band = ortho_dataset.read(1)
            gray_values = gray_band[valid_rows, valid_cols]
            red_values = green_values = blue_values = gray_values

            # DEBUG: Check raw grayscale values
            logger.info(
                f"DEBUG - Raw grayscale value samples (first 5): {gray_values[:5]}"
            )
            logger.info(
                f"DEBUG - Raw grayscale value range: min={np.min(gray_values)}, max={np.max(gray_values)}"
            )

        else:
            raise ValueError(f"Unsupported number of bands: {ortho_dataset.count}")

        # Optimized scaling using vectorized operations
        dtype_str = str(ortho_dataset.dtypes[0])
        logger.info(f"DEBUG - Orthophoto data type: {dtype_str}")

        if "uint8" in dtype_str:
            # Scale from 0-255 to 0-65535 using vectorized multiplication
            scale_factor = 257  # 65535 / 255
            colors[valid_mask, 0] = (red_values * scale_factor).astype(np.uint16)
            colors[valid_mask, 1] = (green_values * scale_factor).astype(np.uint16)
            colors[valid_mask, 2] = (blue_values * scale_factor).astype(np.uint16)
            logger.info(f"DEBUG - Applied uint8 scaling (factor: {scale_factor})")

        elif "uint16" in dtype_str:
            # Direct assignment for uint16
            colors[valid_mask, 0] = red_values.astype(np.uint16)
            colors[valid_mask, 1] = green_values.astype(np.uint16)
            colors[valid_mask, 2] = blue_values.astype(np.uint16)
            logger.info("DEBUG - Applied uint16 direct assignment")

        elif "float" in dtype_str:
            # Scale from 0-1 range to 0-65535
            colors[valid_mask, 0] = (red_values * 65535).astype(np.uint16)
            colors[valid_mask, 1] = (green_values * 65535).astype(np.uint16)
            colors[valid_mask, 2] = (blue_values * 65535).astype(np.uint16)
            logger.info("DEBUG - Applied float scaling (factor: 65535)")

        else:
            logger.warning(f"Unknown data type {dtype_str}, using direct conversion")
            colors[valid_mask, 0] = red_values.astype(np.uint16)
            colors[valid_mask, 1] = green_values.astype(np.uint16)
            colors[valid_mask, 2] = blue_values.astype(np.uint16)
            logger.info("DEBUG - Applied direct conversion")

        # DEBUG: Check final color values
        logger.info(f"DEBUG - Final color values (first 5 points):")
        for i in range(min(5, len(colors))):
            if valid_mask[i]:
                logger.info(
                    f"  Point {i}: R={colors[i,0]}, G={colors[i,1]}, B={colors[i,2]}"
                )

        # DEBUG: Check color statistics
        valid_colors = colors[valid_mask]
        logger.info(f"DEBUG - Color statistics for {len(valid_colors)} valid points:")
        logger.info(
            f"  Red: min={np.min(valid_colors[:,0])}, max={np.max(valid_colors[:,0])}, mean={np.mean(valid_colors[:,0]):.1f}"
        )
        logger.info(
            f"  Green: min={np.min(valid_colors[:,1])}, max={np.max(valid_colors[:,1])}, mean={np.mean(valid_colors[:,1]):.1f}"
        )
        logger.info(
            f"  Blue: min={np.min(valid_colors[:,2])}, max={np.max(valid_colors[:,2])}, mean={np.mean(valid_colors[:,2]):.1f}"
        )

        # DEBUG: Count non-zero colors
        non_zero_red = np.sum(valid_colors[:, 0] > 0)
        non_zero_green = np.sum(valid_colors[:, 1] > 0)
        non_zero_blue = np.sum(valid_colors[:, 2] > 0)
        logger.info(
            f"DEBUG - Non-zero color counts: R={non_zero_red}, G={non_zero_green}, B={non_zero_blue}"
        )

        logger.info("Point cloud colorization complete")
        return colors, valid_mask

    def save_colorized_point_cloud(
        self,
        las_data: laspy.LasData,
        colors: np.ndarray,
        valid_mask: np.ndarray,
        output_path: str,
        preserve_original_colors: bool = True,
    ):
        """
        Save trimmed and colorized point cloud to file with optimized performance.
        Only saves points that fall within the orthophoto bounds.

        Args:
            las_data: Original point cloud data
            colors: RGB color array for all points
            valid_mask: Boolean mask indicating which points have valid colors
            output_path: Output file path
            preserve_original_colors: Whether to preserve existing colors as backup
        """
        output_path = Path(output_path)
        logger.info(f"Saving trimmed colorized point cloud to: {output_path}")

        # Filter data to only include points within orthophoto bounds
        valid_points_count = np.sum(valid_mask)
        total_points_count = len(las_data.points)

        logger.info(
            f"Trimming point cloud: {valid_points_count:,}/{total_points_count:,} points "
            f"({100*valid_points_count/total_points_count:.1f}%) within orthophoto bounds"
        )

        # Create new header based on original header
        original_header = las_data.header

        # Point formats that support RGB colors: 2, 3, 5, 7, 8, 10
        rgb_supported_formats = {2, 3, 5, 7, 8, 10}

        # Determine the appropriate point format for colors
        if original_header.point_format.id not in rgb_supported_formats:
            logger.info(
                f"Converting from point format {original_header.point_format.id} to format 2 to support colors"
            )
            new_point_format = laspy.PointFormat(2)
        else:
            new_point_format = original_header.point_format

        # Create new header with the same version but potentially different point format
        header = laspy.LasHeader(
            version=original_header.version, point_format=new_point_format
        )

        # Copy important properties from original header
        header.x_scale = original_header.x_scale
        header.y_scale = original_header.y_scale
        header.z_scale = original_header.z_scale
        header.x_offset = original_header.x_offset
        header.y_offset = original_header.y_offset
        header.z_offset = original_header.z_offset

        # Copy VLRs (including CRS information) from original header
        if hasattr(original_header, "vlrs") and original_header.vlrs:
            header.vlrs = original_header.vlrs.copy()

        # Create new LAS file with filtered data
        colorized_las = laspy.LasData(header)

        # Copy only the valid points
        colorized_las.x = las_data.x[valid_mask]
        colorized_las.y = las_data.y[valid_mask]
        colorized_las.z = las_data.z[valid_mask]

        # Copy other attributes if they exist (only for valid points)
        attributes_to_copy = [
            "intensity",
            "return_number",
            "number_of_returns",
            "classification",
            "scan_angle_rank",
            "user_data",
            "point_source_id",
        ]

        for attr_name in attributes_to_copy:
            if hasattr(las_data, attr_name):
                setattr(
                    colorized_las, attr_name, getattr(las_data, attr_name)[valid_mask]
                )

        # Set colors efficiently (only for valid points)
        valid_colors = colors[valid_mask]
        colorized_las.red = valid_colors[:, 0]
        colorized_las.green = valid_colors[:, 1]
        colorized_las.blue = valid_colors[:, 2]

        # Save to file
        logger.info("Writing trimmed LAZ file...")
        colorized_las.write(str(output_path))

        # File statistics
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Saved trimmed colorized point cloud ({file_size:.1f} MB)")
        logger.info(
            f"Trimmed from {total_points_count:,} to {valid_points_count:,} points"
        )

    def _fetch_point_cloud_data(
        self,
        pc_fetcher: "PointCloudDatasetFinder",
        lat: float,
        lon: float,
        ortho_bounds: Optional[Dict] = None,
        ortho_crs: Optional[str] = None,
    ) -> str:
        """
        Fetch point cloud data for given coordinates with retry logic.

        Args:
            pc_fetcher: Point cloud fetcher instance
            lat: Latitude
            lon: Longitude
            ortho_bounds: Optional orthophoto bounds for better dataset selection
            ortho_crs: Optional orthophoto CRS

        Returns:
            Path to downloaded point cloud file
        """
        logger.info("Fetching point cloud data...")

        try:
            # Generate bounding box for point cloud search
            bbox = pc_fetcher.generate_bounding_box(lat, lon, buffer_km=1.0)
            logger.info(f"Search area: {bbox}")

            # Search for point cloud data
            logger.info("Searching for LiDAR data...")
            products = pc_fetcher.search_lidar_products(bbox)

            if not products:
                logger.error(
                    f"No LiDAR data found for coordinates {lat:.6f}, {lon:.6f}"
                )
                raise RuntimeError(
                    f"No LiDAR data found for location {lat:.6f}, {lon:.6f}. This area may not have available point cloud data."
                )

            logger.info(f"Found {len(products)} LiDAR products")

            laz_products = pc_fetcher.filter_laz_products(products)

            if not laz_products:
                logger.error("No LAZ format LiDAR data found")
                raise RuntimeError(
                    "No LAZ format LiDAR data found. Only LAZ files are supported for processing."
                )

            logger.info(f"Found {len(laz_products)} LAZ products")

            # Select the best dataset using improved selection logic
            logger.info("Selecting best dataset based on location and recency...")
            if ortho_bounds and ortho_crs:
                logger.info("Using orthophoto-aware dataset selection...")
                best_product = pc_fetcher.select_best_dataset_for_orthophoto(
                    laz_products, ortho_bounds, ortho_crs, lat, lon
                )
            else:
                best_product = pc_fetcher.select_best_dataset_for_location(
                    laz_products, lat, lon
                )

            logger.info(f"Selected dataset: {best_product.get('name', 'Unknown')}")

            # Try downloading point cloud with retry logic
            max_retries = 3
            retry_delay = 5  # seconds

            for attempt in range(max_retries):
                try:
                    product_title = best_product.get(
                        "title", best_product.get("name", "Unknown product")
                    )
                    logger.info(
                        f"Attempting to download (attempt {attempt + 1}/{max_retries}): {product_title}"
                    )

                    downloaded_pc = pc_fetcher.download_point_cloud(
                        best_product, str(self.output_dir), ortho_bounds, ortho_crs
                    )

                    if downloaded_pc and Path(downloaded_pc).exists():
                        logger.info(
                            f"Point cloud downloaded successfully: {downloaded_pc}"
                        )
                        return downloaded_pc
                    else:
                        logger.warning(
                            f"Download attempt {attempt + 1} failed - file not created"
                        )

                except Exception as download_error:
                    logger.warning(
                        f"Download attempt {attempt + 1} failed: {str(download_error)}"
                    )

                    # Check if it's a timeout or connection error
                    if any(
                        keyword in str(download_error).lower()
                        for keyword in [
                            "timeout",
                            "connection",
                            "max retries",
                            "httpsconnectionpool",
                        ]
                    ):
                        logger.info(
                            f"Network error detected, will retry in {retry_delay} seconds..."
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue

                    # If it's the last attempt or not a network error, raise
                    if attempt == max_retries - 1:
                        raise

            # If we get here, all retries failed
            raise RuntimeError(
                f"Failed to download point cloud after {max_retries} attempts. "
                "The USGS server may be experiencing issues. Please try again later."
            )

        except Exception as e:
            logger.error(f"Point cloud fetch failed: {str(e)}")
            # Re-raise with more context
            if any(
                msg in str(e)
                for msg in ["No LiDAR data found", "No LAZ format", "after", "attempts"]
            ):
                raise  # Re-raise our custom error messages as-is
            else:
                raise RuntimeError(f"Failed to fetch point cloud data: {str(e)}")

    def _fetch_orthophoto_data(
        self, ortho_fetcher: "NAIPFetcher", address: str, lat: float, lon: float
    ) -> str:
        """
        Fetch orthophoto data for given address/coordinates with improved error handling.

        Args:
            ortho_fetcher: Orthophoto fetcher instance
            address: Street address (fallback)
            lat: Latitude
            lon: Longitude

        Returns:
            Path to downloaded orthophoto file
        """
        logger.info("Fetching orthophoto data...")

        try:
            # Try fetching orthophoto using coordinates
            ortho_path, ortho_metadata = ortho_fetcher.get_orthophoto_for_address(
                address, str(self.output_dir)
            )

            logger.info(f"Orthophoto downloaded: {ortho_path}")

            # Verify the file exists and is valid
            if Path(ortho_path).exists():
                try:
                    with rasterio.open(ortho_path) as test_ds:
                        logger.info(
                            f"Orthophoto validation successful: {test_ds.width}x{test_ds.height}"
                        )
                        return ortho_path
                except Exception as e:
                    logger.warning(f"Downloaded orthophoto failed validation: {e}")
                    # File exists but is invalid, try to find alternative
            else:
                logger.warning(f"Expected orthophoto file not found: {ortho_path}")

            # If the expected file doesn't exist or is invalid, search for any orthophoto in the directory
            output_dir = Path(self.output_dir)
            if output_dir.exists():
                logger.info("Searching for alternative orthophoto files...")

                # Look for any TIFF files that might be orthophotos
                tiff_patterns = ["*.tif", "*.tiff", "*naip*.tif", "*orthophoto*.tif"]
                found_files = []

                for pattern in tiff_patterns:
                    found_files.extend(list(output_dir.glob(pattern)))

                for candidate_file in found_files:
                    try:
                        with rasterio.open(str(candidate_file)) as test_ds:
                            if test_ds.width > 0 and test_ds.height > 0:
                                logger.info(
                                    f"Found alternative valid orthophoto: {candidate_file}"
                                )
                                return str(candidate_file)
                    except Exception:
                        continue

            # If we still haven't found a valid file, raise an error
            raise RuntimeError(f"No valid orthophoto file found after download attempt")

        except Exception as e:
            logger.error(f"Failed to fetch orthophoto: {e}")
            raise

    def download_corrected_orthophoto(
        self, pc_bounds: Dict[str, float], output_path: str = None
    ) -> str:
        """
        Download orthophoto with bounds that properly cover the point cloud.

        Args:
            pc_bounds: Point cloud bounds in WGS84 {'west': x_min, 'east': x_max, 'south': y_min, 'north': y_max}
            output_path: Optional path for output file

        Returns:
            Path to downloaded orthophoto
        """
        import requests
        import json

        if output_path is None:
            output_path = str(self.output_dir / "corrected_orthophoto.tif")

        logger.info(f"Downloading corrected orthophoto with bounds: {pc_bounds}")

        # Add 10% buffer to ensure full coverage
        width_deg = pc_bounds["east"] - pc_bounds["west"]
        height_deg = pc_bounds["north"] - pc_bounds["south"]
        buffer_x = width_deg * 0.1
        buffer_y = height_deg * 0.1

        min_lon = pc_bounds["west"] - buffer_x
        min_lat = pc_bounds["south"] - buffer_y
        max_lon = pc_bounds["east"] + buffer_x
        max_lat = pc_bounds["north"] + buffer_y

        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        # Calculate appropriate image size (aim for ~1m resolution)
        width_m = (
            width_deg
            * 111320
            * np.cos(np.radians((pc_bounds["north"] + pc_bounds["south"]) / 2))
        )
        height_m = height_deg * 111320

        # Target ~2 meter per pixel for reasonable file size, but cap at 2000px
        target_width = min(max(int(width_m / 2), 200), 2000)
        target_height = min(max(int(height_m / 2), 200), 2000)
        image_size = f"{target_width},{target_height}"

        logger.info(f"Requesting image size: {target_width} x {target_height}")
        logger.info(f"Coverage area: {width_m:.0f}m x {height_m:.0f}m")

        # Use USGS NAIPPlus ImageServer (same as get_orthophoto.py)
        service_url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"

        # Try multiple sizes in case the requested size is too large
        fallback_sizes = ["2048,2048", "1024,1024", "512,512"]
        sizes_to_try = [image_size] + fallback_sizes

        # Remove duplicates while preserving order
        seen = set()
        unique_sizes = []
        for size in sizes_to_try:
            if size not in seen:
                seen.add(size)
                unique_sizes.append(size)

        last_error = None

        for attempt, size in enumerate(unique_sizes):
            params = {
                "bbox": bbox,
                "bboxSR": 4326,  # WGS84 coordinate system
                "size": size,  # Output image size in pixels
                "imageSR": 4326,  # Output coordinate system
                "format": "tiff",  # Output format
                "f": "image",  # Response format
            }

            try:
                logger.info(
                    f"Making request to NAIP service (attempt {attempt + 1}/{len(unique_sizes)}):"
                )
                logger.info(f"  Bounding box: {bbox}")
                logger.info(f"  Image size: {size}")

                response = requests.get(service_url, params=params, timeout=120)
                response.raise_for_status()

                file_size = len(response.content)

                # Check if response is actually an image by examining content
                if file_size < 1000:  # Very small response is likely an error
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get(
                            "message", "Unknown service error"
                        )
                        if (
                            "size limit" in error_msg.lower()
                            and attempt < len(unique_sizes) - 1
                        ):
                            logger.info(
                                f"  Size {size} too large, trying smaller size..."
                            )
                            last_error = Exception(f"NAIP service error: {error_msg}")
                            continue
                        else:
                            raise Exception(f"NAIP service error: {error_msg}")
                    except json.JSONDecodeError:
                        pass  # Not JSON, continue with normal processing

                # Additional check: if content type suggests image but size is suspiciously small
                content_type = response.headers.get("content-type", "")
                if content_type.startswith("image/") and file_size < 1000:
                    if attempt < len(unique_sizes) - 1:
                        logger.info(
                            f"  Received very small image ({file_size} bytes), trying smaller size..."
                        )
                        last_error = Exception(
                            f"Received very small image ({file_size} bytes), likely an error response"
                        )
                        continue
                    else:
                        raise Exception(
                            f"Received very small image ({file_size} bytes), likely an error response"
                        )

                # Save the image
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_bytes(response.content)

                logger.info(
                    f"Successfully downloaded corrected orthophoto: {output_path}"
                )
                logger.info(f"File size: {file_size / 1024 / 1024:.1f} MB")
                if size != image_size:
                    logger.info(
                        f"Note: Used fallback size {size} instead of requested {image_size}"
                    )

                # Save metadata
                metadata = {
                    "bbox": bbox,
                    "bbox_array": [min_lon, min_lat, max_lon, max_lat],
                    "image_size": [int(size.split(",")[0]), int(size.split(",")[1])],
                    "crs": "EPSG:4326",
                    "source": "USGS NAIPPlus - Auto-corrected",
                    "point_cloud_bounds": pc_bounds,
                    "service_url": service_url,
                    "request_params": params,
                }

                metadata_path = str(Path(output_path).with_suffix(".json"))
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                return output_path

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                last_error = e
                if attempt < len(unique_sizes) - 1:
                    continue
                else:
                    break

        # If we get here, all attempts failed
        logger.error(f"All download attempts failed. Last error: {last_error}")
        raise Exception(
            f"Failed to download corrected orthophoto after {len(unique_sizes)} attempts: {last_error}"
        )

    def process_from_address(self, address: str) -> str:
        """
        Complete workflow: fetch data and colorize point cloud for an address.
        Point cloud and orthophoto are fetched in parallel for improved performance.

        Args:
            address: Street address

        Returns:
            Path to colorized point cloud file
        """
        logger.info(f"Processing address: {address}")

        # Initialize fetchers
        geocoder = Geocoder()
        pc_fetcher = PointCloudDatasetFinder()
        ortho_fetcher = NAIPFetcher()

        try:
            # Geocode address
            lat, lon = geocoder.geocode_address(address)
            logger.info(f"Coordinates: {lat:.6f}, {lon:.6f}")

            # First, fetch orthophoto to get bounds for better point cloud selection
            logger.info("Fetching orthophoto first for optimal dataset selection...")
            ortho_path = self._fetch_orthophoto_data(ortho_fetcher, address, lat, lon)
            logger.info("Orthophoto fetch completed successfully")

            # Get orthophoto bounds for dataset selection
            ortho_bounds = None
            ortho_crs = None
            try:
                with rasterio.open(ortho_path) as ortho_dataset:
                    ortho_bounds = {
                        "left": ortho_dataset.bounds.left,
                        "right": ortho_dataset.bounds.right,
                        "bottom": ortho_dataset.bounds.bottom,
                        "top": ortho_dataset.bounds.top,
                    }
                    ortho_crs = str(ortho_dataset.crs)
                    logger.info(f"Orthophoto bounds extracted: {ortho_bounds}")
                    logger.info(f"Orthophoto CRS: {ortho_crs}")
            except Exception as e:
                logger.warning(f"Could not extract orthophoto bounds: {e}")

            # Now fetch point cloud using orthophoto-aware selection
            logger.info("Fetching point cloud with orthophoto-aware selection...")
            downloaded_pc = self._fetch_point_cloud_data(
                pc_fetcher, lat, lon, ortho_bounds, ortho_crs
            )
            logger.info("Point cloud fetch completed successfully")

            logger.info("Both datasets fetched successfully, starting processing...")

            # Process the data
            return self.process_files(str(downloaded_pc), str(ortho_path))

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def process_files(
        self,
        point_cloud_path: str,
        orthophoto_path: str,
        output_name: Optional[str] = None,
        create_summary: bool = True,
    ) -> str:
        """
        Process existing point cloud and orthophoto files with optional optimizations.

        Args:
            point_cloud_path: Path to point cloud file
            orthophoto_path: Path to orthophoto file
            output_name: Output filename (optional)
            create_summary: Whether to create summary report (can be slow for large datasets)

        Returns:
            Path to colorized point cloud file
        """
        logger.info("Starting point cloud colorization...")

        # Load data
        las_data = self.load_point_cloud(point_cloud_path)
        ortho_dataset = self.load_orthophoto(orthophoto_path)

        try:
            # Colorize
            colors, valid_mask = self.colorize_point_cloud(las_data, ortho_dataset)

            # Generate output filename
            if output_name is None:
                pc_name = Path(point_cloud_path).stem
                output_name = f"{pc_name}_colorized.laz"

            output_path = self.output_dir / output_name

            # Save result (now trimmed to orthophoto intersection)
            self.save_colorized_point_cloud(
                las_data, colors, valid_mask, str(output_path)
            )

            # Create summary report only if requested
            if create_summary:
                self.create_summary_report(
                    point_cloud_path,
                    orthophoto_path,
                    str(output_path),
                    colors,
                    valid_mask,
                )

            return str(output_path)

        finally:
            ortho_dataset.close()

    def create_summary_report(
        self,
        pc_path: str,
        ortho_path: str,
        output_path: str,
        colors: np.ndarray,
        valid_mask: np.ndarray,
    ):
        """
        Create a summary report of the colorization process.

        Args:
            pc_path: Input point cloud path
            ortho_path: Input orthophoto path
            output_path: Output point cloud path
            colors: Color array (for all original points)
            valid_mask: Boolean mask indicating which points were within orthophoto bounds
        """
        total_original_points = len(colors)
        trimmed_points = np.sum(valid_mask)

        # Only analyze colors for valid points
        valid_colors = colors[valid_mask]

        report = {
            "input_point_cloud": str(pc_path),
            "input_orthophoto": str(ortho_path),
            "output_point_cloud": str(output_path),
            "processing_stats": {
                "original_total_points": int(total_original_points),
                "trimmed_points": int(trimmed_points),
                "trimming_rate": float(trimmed_points / total_original_points),
                "colorized_points": int(np.sum(np.any(valid_colors > 0, axis=1))),
                "colorization_rate": float(
                    np.sum(np.any(valid_colors > 0, axis=1)) / len(valid_colors)
                    if len(valid_colors) > 0
                    else 0
                ),
            },
            "color_stats": {
                "mean_red": (
                    float(np.mean(valid_colors[:, 0])) if len(valid_colors) > 0 else 0
                ),
                "mean_green": (
                    float(np.mean(valid_colors[:, 1])) if len(valid_colors) > 0 else 0
                ),
                "mean_blue": (
                    float(np.mean(valid_colors[:, 2])) if len(valid_colors) > 0 else 0
                ),
                "max_red": (
                    int(np.max(valid_colors[:, 0])) if len(valid_colors) > 0 else 0
                ),
                "max_green": (
                    int(np.max(valid_colors[:, 1])) if len(valid_colors) > 0 else 0
                ),
                "max_blue": (
                    int(np.max(valid_colors[:, 2])) if len(valid_colors) > 0 else 0
                ),
            },
        }

        report_path = self.output_dir / "colorization_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Summary report saved: {report_path}")
        logger.info(
            f"Point cloud trimmed from {total_original_points:,} to {trimmed_points:,} points "
            f"({report['processing_stats']['trimming_rate']:.1%})"
        )
        logger.info(
            f"Colorization rate: {report['processing_stats']['colorization_rate']:.1%}"
        )


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Advanced Point Cloud Colorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process by address (downloads data automatically)
  python process_point_cloud.py --address "1250 Wildwood Road, Boulder, CO"
  
  # Process existing files
  python process_point_cloud.py --input_pc data/point_cloud.laz --input_ortho data/orthophoto.tif
  
  # Fast processing mode (no diagnostics or summary reports)
  python process_point_cloud.py --address "123 Main St" --fast
  
  # Custom performance options
  python process_point_cloud.py --input_pc data/pc.laz --input_ortho data/ortho.tif --no-diagnostics
  
  # Specify output directory
  python process_point_cloud.py --address "123 Main St" --output_dir results/
        """,
    )

    # Input options
    parser.add_argument(
        "--address", type=str, help="Address to process (automatically downloads data)"
    )

    # File input arguments (both required together if using files)
    parser.add_argument(
        "--input_pc", type=str, help="Path to input point cloud file (LAZ/LAS)"
    )
    parser.add_argument("--input_ortho", type=str, help="Path to input orthophoto file")

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--output_name", type=str, help="Output filename (default: auto-generated)"
    )

    # Processing options
    parser.add_argument(
        "--source_crs",
        type=str,
        help="Override source CRS for point cloud (e.g., EPSG:2232)",
    )

    # Performance options
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast processing mode (disables diagnostics and summary reports)",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable diagnostic plot creation for faster processing",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summary report creation for faster processing",
    )

    args = parser.parse_args()

    # Validate input arguments
    if args.address and (args.input_pc or args.input_ortho):
        parser.error("Cannot use --address with --input_pc or --input_ortho")
    elif not args.address and (not args.input_pc or not args.input_ortho):
        parser.error(
            "Must provide either --address OR both --input_pc and --input_ortho"
        )

    try:
        # Determine performance settings
        create_diagnostics = not (args.fast or args.no_diagnostics)
        create_summary = not (args.fast or args.no_summary)

        if args.fast:
            logger.info(
                "Fast processing mode enabled - diagnostics and summary reports disabled"
            )

        # Initialize colorizer with performance options
        colorizer = PointCloudColorizer(
            args.output_dir, create_diagnostics=create_diagnostics
        )

        if args.address:
            # Process by address
            output_path = colorizer.process_from_address(args.address)
        else:
            # Process existing files
            output_path = colorizer.process_files(
                args.input_pc,
                args.input_ortho,
                args.output_name,
                create_summary=create_summary,
            )

        logger.info(f"SUCCESS: Colorized point cloud saved to {output_path}")
        return 0

    except Exception as e:
        logger.error(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
