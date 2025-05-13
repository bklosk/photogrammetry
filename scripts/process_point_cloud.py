#!/usr/bin/env python3
"""
Point Cloud Processing Script: Colorization and Visualization

This script provides functionalities to:
1. Colorize a LAZ/LAS point cloud using satellite imagery.
2. Visualize LAZ/LAS point cloud files using PyVista.
3. Generate a mesh from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer

# Try to import laspy for reading LAZ files
try:
    import laspy
except ImportError:
    print("ERROR: laspy could not be imported.")
    print("Please install laspy: pip install laspy lazrs")
    sys.exit(1)

# Try to import PyVista for visualization
VISUALIZATION_LIB = None
try:
    import pyvista as pv

    VISUALIZATION_LIB = "pyvista"
except ImportError:
    print("WARNING: PyVista could not be imported.")
    print("Visualization and mesh generation will not be available.")
    print("Please install PyVista: pip install pyvista")


# --- Point Cloud and Satellite Image Loading ---
def load_point_cloud(file_path):
    """
    Load a LAZ/LAS point cloud file.
    """
    if not os.path.exists(file_path):
        print(f"Error: Point cloud file not found: {file_path}")
        sys.exit(1)
    try:
        las_data = laspy.read(file_path)
        return las_data
    except Exception as e:
        print(f"Error loading point cloud: {str(e)}")
        sys.exit(1)


def load_satellite_image(file_path):
    """
    Load a satellite image using rasterio.
    """
    if not os.path.exists(file_path):
        print(f"Error: Satellite image file not found: {file_path}")
        sys.exit(1)
    try:
        dataset = rasterio.open(file_path)
        return dataset
    except Exception as e:
        print(f"Error loading satellite image: {str(e)}")
        sys.exit(1)


# --- CRS and Transformation Functions ---
def identify_point_cloud_crs(las_data):
    """
    Attempt to identify the coordinate reference system of the point cloud.
    """
    if (
        hasattr(las_data, "header")
        and hasattr(las_data.header, "crs")
        and las_data.header.crs
    ):
        return str(las_data.header.crs)

    print(
        "Point cloud CRS not found in header. Attempting to use assumed EPSG:2232 (NAD83 Colorado Central ftUS)."
    )
    return "EPSG:2232"  # Default assumption


def transform_coordinates(las_data, target_crs_str, custom_transform_func=None):
    """
    Transform point cloud coordinates to the target CRS.
    """
    x_coords = las_data.x
    y_coords = las_data.y

    source_crs_str = identify_point_cloud_crs(las_data)

    if source_crs_str:
        try:
            transformer = Transformer.from_crs(
                source_crs_str, target_crs_str, always_xy=True
            )
            transformed_x, transformed_y = transformer.transform(x_coords, y_coords)
            return np.column_stack((transformed_x, transformed_y))
        except Exception as e:
            print(f"Error during pyproj transformation: {e}")
            print(
                "Falling back to custom transformation if available, or simple scaling."
            )

    if custom_transform_func:
        transformed_x, transformed_y = custom_transform_func(x_coords, y_coords)
        return np.column_stack((transformed_x, transformed_y))
    else:
        print(
            "WARNING: No valid CRS identified for point cloud and no custom transform provided."
        )
        print(
            "Falling back to approximate bounds-to-bounds scaling. Results may be inaccurate."
        )
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        # These are example satellite image bounds in EPSG:26913 (UTM 13N)
        # TODO: Make these configurable or derive from raster_dataset if possible
        target_min_x, target_max_x = 476216.9806, 477833.0806
        target_min_y, target_max_y = 4424864.8564, 4426481.1064

        x_scale = (
            (target_max_x - target_min_x) / (max_x - min_x)
            if (max_x - min_x) != 0
            else 1
        )
        y_scale = (
            (target_max_y - target_min_y) / (max_y - min_y)
            if (max_y - min_y) != 0
            else 1
        )

        transformed_x = target_min_x + (x_coords - min_x) * x_scale
        transformed_y = target_min_y + (y_coords - min_y) * y_scale
        return np.column_stack((transformed_x, transformed_y))


def define_custom_transform():
    """
    Define a fallback custom transformation function.
    """

    def transform_func(x, y):
        # Point cloud bounds (source) - Example, should be adjusted
        source_min_x, source_max_x = 3065262.93, 3065728.85
        source_min_y, source_max_y = 1780508.57, 1780808.63

        # Satellite image bounds (target) in UTM Zone 13N - Example
        target_min_x, target_max_x = 476216.9806, 477833.0806
        target_min_y, target_max_y = 4424864.8564, 4426481.1064

        x_scale = (target_max_x - target_min_x) / (source_max_x - source_min_x)
        y_scale = (target_max_y - target_min_y) / (source_max_y - source_min_y)
        x_offset = target_min_x - source_min_x * x_scale
        y_offset = target_min_y - source_min_y * y_scale

        new_x = x * x_scale + x_offset
        new_y = y * y_scale + y_offset
        return new_x, new_y

    return transform_func


# --- Colorization Core Functions ---
def convert_point_cloud_to_image_coordinates(
    las_data, raster_dataset, custom_transform_func=None, x_offset=0.0, y_offset=0.0
):
    """
    Convert point cloud coordinates to image pixel coordinates.
    """
    coords = transform_coordinates(
        las_data, str(raster_dataset.crs), custom_transform_func
    )

    # Apply user-defined offsets
    if x_offset != 0.0 or y_offset != 0.0:
        coords[:, 0] += x_offset
        coords[:, 1] += y_offset

    rows, cols = rasterio.transform.rowcol(
        raster_dataset.transform, coords[:, 0], coords[:, 1]
    )
    pixel_x = np.array(cols)
    pixel_y = np.array(rows)

    valid_points = np.logical_and.reduce(
        (
            pixel_x >= 0,
            pixel_x < raster_dataset.width,
            pixel_y >= 0,
            pixel_y < raster_dataset.height,
        )
    )

    if np.sum(valid_points) == 0:
        print(
            "WARNING: No points fall within the image bounds. Check CRS and image coverage."
        )
        # Consider saving a diagnostic plot here as in the original colorize_with_satellite.py
        fig, ax = plt.subplots(figsize=(10, 8))
        img_bounds = raster_dataset.bounds
        rect = plt.Rectangle(
            (img_bounds.left, img_bounds.bottom),
            img_bounds.right - img_bounds.left,
            img_bounds.top - img_bounds.bottom,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
            label="Satellite Image Bounds",
        )
        ax.add_patch(rect)
        sample_size = min(1000, len(coords))
        sample_indices = np.random.choice(len(coords), sample_size, replace=False)
        ax.scatter(
            coords[sample_indices, 0],
            coords[sample_indices, 1],
            s=1,
            c="blue",
            alpha=0.5,
            label="Transformed Point Cloud Points",
        )
        ax.set_title("Transformed Point Cloud vs. Satellite Image Bounds")
        ax.set_xlabel(f"X Coordinate ({raster_dataset.crs})")
        ax.set_ylabel(f"Y Coordinate ({raster_dataset.crs})")
        ax.legend()
        ax.grid(True)
        plt.savefig("coordinate_mismatch_diagnostic.png")
        plt.close()
        print("Diagnostic plot saved to 'coordinate_mismatch_diagnostic.png'")

    return pixel_x, pixel_y, valid_points


def extract_colors_from_image(raster_dataset, pixel_x, pixel_y, valid_points):
    """
    Extract colors from the satellite image.
    """
    valid_x_indices = pixel_x[valid_points]  # Renamed for clarity
    valid_y_indices = pixel_y[valid_points]  # Renamed for clarity

    num_total_points = len(pixel_x)
    colors_out = np.zeros((num_total_points, 3), dtype=np.uint16)

    if np.sum(valid_points) == 0:
        print("No valid points to colorize.")
        return colors_out

    bands = raster_dataset.count
    source_dtype_str = raster_dataset.dtypes[0]  # Assuming all bands same dtype

    # Initialize float arrays for extracted colors for valid points
    num_valid_points = np.sum(valid_points)
    r_float = np.zeros(num_valid_points, dtype=np.float32)
    g_float = np.zeros(num_valid_points, dtype=np.float32)
    b_float = np.zeros(num_valid_points, dtype=np.float32)

    if bands >= 3:
        r_band_data = raster_dataset.read(1)
        g_band_data = raster_dataset.read(2)
        b_band_data = raster_dataset.read(3)
        r_float = r_band_data[valid_y_indices, valid_x_indices].astype(np.float32)
        g_float = g_band_data[valid_y_indices, valid_x_indices].astype(np.float32)
        b_float = b_band_data[valid_y_indices, valid_x_indices].astype(np.float32)
    elif bands == 1:  # Grayscale
        gray_band_data = raster_dataset.read(1)
        gray_float = gray_band_data[valid_y_indices, valid_x_indices].astype(np.float32)
        r_float, g_float, b_float = gray_float, gray_float, gray_float
    else:
        print(
            f"Unsupported number of bands in satellite image: {bands}. Cannot extract color."
        )
        return colors_out  # Return zeros

    # Default to no scaling if not explicitly handled
    scaled_r, scaled_g, scaled_b = r_float, g_float, b_float

    if source_dtype_str == "uint8":
        scaled_r = r_float / 255.0 * 65535.0
        scaled_g = g_float / 255.0 * 65535.0
        scaled_b = b_float / 255.0 * 65535.0
    elif source_dtype_str == "uint16":
        pass
    elif source_dtype_str.startswith("float"):
        # Check actual range for warning, but scale assuming 0-1 for robustness
        min_val_check = min(
            np.min(r_float) if r_float.size > 0 else 0.0,
            np.min(g_float) if g_float.size > 0 else 0.0,
            np.min(b_float) if b_float.size > 0 else 0.0,
        )
        max_val_check = max(
            np.max(r_float) if r_float.size > 0 else 1.0,
            np.max(g_float) if g_float.size > 0 else 1.0,
            np.max(b_float) if b_float.size > 0 else 1.0,
        )
        # Allow all zeros to pass without warning if that's the data
        is_all_zero = max_val_check == 0.0 and min_val_check == 0.0

        if not is_all_zero and not (
            0.0 <= min_val_check <= 1.0 and 0.0 <= max_val_check <= 1.0
        ):
            print(
                f"WARNING: Float data may be outside expected 0.0-1.0 range (min: {min_val_check}, max: {max_val_check}). Resulting colors might be clipped or unexpected."
            )

        scaled_r = r_float * 65535.0
        scaled_g = g_float * 65535.0
        scaled_b = b_float * 65535.0
    else:
        print(
            f"Source dtype {source_dtype_str} is not uint8, uint16, or float. Applying heuristic scaling based on max value."
        )
        current_max_val = 0
        if r_float.size > 0:
            current_max_val = max(current_max_val, np.max(r_float))
        if g_float.size > 0:
            current_max_val = max(current_max_val, np.max(g_float))
        if b_float.size > 0:
            current_max_val = max(current_max_val, np.max(b_float))

        if current_max_val > 0 and current_max_val < 256:
            scaled_r = r_float / 255.0 * 65535.0
            scaled_g = g_float / 255.0 * 65535.0
            scaled_b = b_float / 255.0 * 65535.0
        else:
            pass

    colors_out[valid_points, 0] = np.clip(scaled_r, 0, 65535).astype(np.uint16)
    colors_out[valid_points, 1] = np.clip(scaled_g, 0, 65535).astype(np.uint16)
    colors_out[valid_points, 2] = np.clip(scaled_b, 0, 65535).astype(np.uint16)

    return colors_out


def save_colorized_point_cloud(input_las_data, colors, output_path):
    """
    Save a new point cloud with the extracted colors.
    """

    # Create new LAS data with a point format that includes color
    header = input_las_data.header
    colorized_las = laspy.LasData(header)
    colorized_las.points = input_las_data.points.copy()  # Copy all point data

    # Set the new colors
    colorized_las.red = colors[:, 0]
    colorized_las.green = colors[:, 1]
    colorized_las.blue = colors[:, 2]

    colorized_las.write(output_path)


# --- Visualization Functions ---
def visualize_with_pyvista(las_data, window_title="Point Cloud Viewer"):
    """
    Visualize point cloud with PyVista.
    """
    if VISUALIZATION_LIB != "pyvista":
        print("PyVista not available or not selected.")
        return

    points = np.vstack((las_data.x, las_data.y, las_data.z)).T
    cloud = pv.PolyData(points)

    plotter = pv.Plotter(window_size=[1024, 768], title=window_title)

    if (
        hasattr(las_data, "red")
        and hasattr(las_data, "green")
        and hasattr(las_data, "blue")
    ):
        # Scale 16-bit colors (0-65535) to 8-bit (0-255) for PyVista's RGB
        r_8bit = np.round(las_data.red / 65535.0 * 255).astype(np.uint8)
        g_8bit = np.round(las_data.green / 65535.0 * 255).astype(np.uint8)
        b_8bit = np.round(las_data.blue / 65535.0 * 255).astype(np.uint8)

        rgb_colors_uint8 = np.vstack((r_8bit, g_8bit, b_8bit)).T

        plotter.add_points(
            cloud,
            scalars=rgb_colors_uint8,
            rgb=True,
            render_points_as_spheres=True,
            point_size=3,
        )
    else:
        plotter.add_points(cloud, render_points_as_spheres=True, point_size=3)

    plotter.add_axes()
    plotter.set_background([0.1, 0.1, 0.1])

    plotter.show()


# --- Mesh Generation Function ---
def generate_mesh_from_point_cloud(point_cloud_path, output_mesh_path):
    """
    Generates a mesh from a point cloud using PyVista.
    """
    if not pv:  # Check if pv (PyVista) was imported
        print("PyVista is required for mesh generation. Please install PyVista.")
        try:
            import pyvista as pv_local
        except ImportError:
            print("Failed to import PyVista for mesh generation.")
            return
    else:
        pv_local = pv  # Use the globally imported pv

    try:
        las_data = load_point_cloud(point_cloud_path)  # Use our loader
        points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
        pv_cloud = pv_local.PointCloud(points)

        if (
            hasattr(las_data, "red")
            and hasattr(las_data, "green")
            and hasattr(las_data, "blue")
        ):
            red = np.round(las_data.red / 65535.0 * 255).astype(np.uint8)
            green = np.round(las_data.green / 65535.0 * 255).astype(np.uint8)
            blue = np.round(las_data.blue / 65535.0 * 255).astype(np.uint8)
            pv_cloud.point_data["colors"] = np.vstack((red, green, blue)).transpose()
            pv_cloud.active_scalars_name = "colors"

        # Note: Delaunay 3D might not be suitable for all point clouds (e.g., non-convex, noisy)
        # Other methods like pv.reconstruct_surface() (Poisson) could be alternatives but have more dependencies.
        surf = pv_cloud.delaunay_3d().extract_surface()

        if surf.n_points == 0 or surf.n_cells == 0:
            print(
                "Mesh generation resulted in an empty mesh. This can happen if the point cloud is too sparse or planar for Delaunay 3D."
            )
            print(
                "Consider trying a different meshing algorithm or checking point cloud density."
            )
            return

        surf.save(output_mesh_path, binary=True)

    except Exception as e:
        print(f"Error generating mesh: {str(e)}")
        import traceback

        traceback.print_exc()


# --- Argument Parsing ---
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process point clouds: colorize with satellite imagery and/or visualize.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Mode of operation"
    )

    # Visualize mode
    parser_visualize = subparsers.add_parser(
        "visualize", help="Visualize an existing point cloud file."
    )
    parser_visualize.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to the point cloud file (LAZ/LAS) to visualize.",
    )
    parser_visualize.add_argument(
        "--lib",
        type=str,
        choices=["pyvista"],
        default=VISUALIZATION_LIB,
        help=f"Preferred visualization library (default: {VISUALIZATION_LIB if VISUALIZATION_LIB else 'auto-detect'}). Only PyVista is currently supported.",
    )

    # Colorize mode
    parser_colorize = subparsers.add_parser(
        "colorize", help="Colorize a point cloud using satellite imagery."
    )
    parser_colorize.add_argument(
        "--input_pc",
        type=str,
        required=True,
        help="Path to the input point cloud file (LAZ/LAS).",
    )
    parser_colorize.add_argument(
        "--input_sat",
        type=str,
        required=True,
        help="Path to the satellite image file (e.g., GeoTIFF).",
    )
    parser_colorize.add_argument(
        "--output_pc",
        type=str,
        required=True,
        help="Path to save the colorized point cloud file (LAZ/LAS).",
    )
    parser_colorize.add_argument(
        "--use_custom_transform",
        action="store_true",
        help="Use the fallback custom transformation (bounds scaling) instead of pyproj or if pyproj fails.",
    )
    parser_colorize.add_argument(
        "--x_offset",
        type=float,
        default=0.0,
        help="Apply a manual X offset (in the satellite image CRS units) to the point cloud coordinates before color sampling. Positive values typically shift sampling eastward.",
    )
    parser_colorize.add_argument(
        "--y_offset",
        type=float,
        default=0.0,
        help="Apply a manual Y offset (in the satellite image CRS units) to the point cloud coordinates before color sampling. Positive values typically shift sampling northward.",
    )
    parser_colorize.add_argument(
        "--generate_mesh",
        type=str,
        default=None,
        help="Optional: Path to save a generated mesh (e.g., .ply, .stl) from the colorized point cloud. Requires PyVista.",
    )
    parser_colorize.add_argument(
        "--visualize_result",
        action="store_true",
        help="Visualize the colorized point cloud after processing.",
    )
    parser_colorize.add_argument(
        "--vis_lib_after_colorize",
        type=str,
        choices=["pyvista"],
        default=VISUALIZATION_LIB,
        help=f"Preferred visualization library for '--visualize_result' (default: {VISUALIZATION_LIB if VISUALIZATION_LIB else 'auto-detect'}). Only PyVista is currently supported.",
    )

    return parser.parse_args()


# --- Main Function ---
def main():
    args = parse_arguments()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    global VISUALIZATION_LIB  # Allow modification if user specifies a lib

    if args.mode == "visualize":
        if not VISUALIZATION_LIB and not args.lib:
            print("No visualization library available or specified. Cannot visualize.")
            return 1
        if args.lib:  # User specified a library
            if args.lib == "pyvista" and pv:
                VISUALIZATION_LIB = "pyvista"
            elif args.lib == "pyvista" and not pv:
                print("PyVista specified but not available. Please install PyVista.")
                return 1
            else:  # Should not happen with current choices
                print(f"Unsupported library specified: {args.lib}")
                return 1
        elif not VISUALIZATION_LIB and pv:  # Auto-detect if not specified
            VISUALIZATION_LIB = "pyvista"

        pc_path = (
            args.file
            if os.path.isabs(args.file)
            else os.path.join(script_dir, args.file)
        )
        las_data = load_point_cloud(pc_path)

        if VISUALIZATION_LIB == "pyvista":
            visualize_with_pyvista(
                las_data, window_title=f"Visualize: {os.path.basename(pc_path)}"
            )
        else:
            print(f"PyVista is not available. Cannot visualize.")
            return 1

    elif args.mode == "colorize":
        input_pc_path = (
            args.input_pc
            if os.path.isabs(args.input_pc)
            else os.path.join(script_dir, args.input_pc)
        )
        input_sat_path = (
            args.input_sat
            if os.path.isabs(args.input_sat)
            else os.path.join(script_dir, args.input_sat)
        )
        output_pc_path = (
            args.output_pc
            if os.path.isabs(args.output_pc)
            else os.path.join(script_dir, args.output_pc)
        )

        las_data = load_point_cloud(input_pc_path)
        raster_dataset = load_satellite_image(input_sat_path)

        custom_transform_to_use = None
        if args.use_custom_transform:
            custom_transform_to_use = define_custom_transform()

        pixel_x, pixel_y, valid_points = convert_point_cloud_to_image_coordinates(
            las_data,
            raster_dataset,
            custom_transform_to_use,
            args.x_offset,
            args.y_offset,
        )

        if np.sum(valid_points) == 0:
            print(
                "Colorization cannot proceed as no points are within the image bounds."
            )
            raster_dataset.close()
            return 1

        colors = extract_colors_from_image(
            raster_dataset, pixel_x, pixel_y, valid_points
        )
        save_colorized_point_cloud(las_data, colors, output_pc_path)
        raster_dataset.close()

        if args.generate_mesh:
            output_mesh_filepath = (
                args.generate_mesh
                if os.path.isabs(args.generate_mesh)
                else os.path.join(script_dir, args.generate_mesh)
            )
            generate_mesh_from_point_cloud(output_pc_path, output_mesh_filepath)

        if args.visualize_result:
            if not VISUALIZATION_LIB and not args.vis_lib_after_colorize:
                print(
                    "No visualization library available or specified. Cannot visualize result."
                )
            else:
                if args.vis_lib_after_colorize:
                    VISUALIZATION_LIB = args.vis_lib_after_colorize
                elif not VISUALIZATION_LIB and pv:  # Auto-detect if not specified
                    VISUALIZATION_LIB = "pyvista"

                colorized_las_data = load_point_cloud(output_pc_path)
                if VISUALIZATION_LIB == "pyvista":
                    visualize_with_pyvista(
                        colorized_las_data,
                        window_title=f"Colorized: {os.path.basename(output_pc_path)}",
                    )
                else:
                    print(f"PyVista is not available for visualizing the result.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
