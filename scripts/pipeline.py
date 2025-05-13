#!/usr/bin/env python3
"""
Photogrammetry Pipeline Script

This script orchestrates the point cloud processing pipeline:
1. Colorizes a point cloud using satellite imagery.
2. Visualizes the resulting colorized point cloud.

It replaces the functionality of run_app.sh with hardcoded paths
and no command-line arguments.
"""
import os
import sys

try:
    # Assuming process_point_cloud.py is in the same directory (scripts/)
    import process_point_cloud as ppc
except ImportError as e:
    print(f"Error: Could not import 'process_point_cloud.py'. Ensure it is in the same directory as this script or in the PYTHONPATH.")
    print(f"Details: {e}")
    # Attempt to add the script's directory to sys.path as a fallback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        try:
            import process_point_cloud as ppc
            print("Successfully imported 'process_point_cloud' after adding script directory to sys.path.")
        except ImportError as e_retry:
            print(f"Retry import failed: {e_retry}")
            sys.exit(1)
    else:
        sys.exit(1)


def run_photogrammetry_pipeline():
    """
    Executes the hardcoded photogrammetry pipeline: colorization then visualization.
    """
    # Determine the base directory of this script (scripts/)
    # All paths are relative to the 'scripts' directory, going up to 'data/'
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Hardcoded file paths (mirroring run_app.sh)
    # os.path.normpath is used to ensure paths are in the correct format for the OS
    input_pc_path = os.path.normpath(os.path.join(base_dir, "../data/point_cloud.laz"))
    input_sat_path = os.path.normpath(os.path.join(base_dir, "../data/boulder_flyover/images/n2w235.tif"))
    output_pc_path = os.path.normpath(os.path.join(base_dir, "../data/colorized_point_cloud.laz"))

    # Hardcoded colorization parameters (from run_app.sh)
    x_offset = -2.0
    y_offset = 0.0

    print("Starting photogrammetry pipeline...")

    # --- 1. Colorization Phase ---
    print("\n--- Colorization Phase ---")
    print(f"Loading input point cloud: {input_pc_path}")
    # Functions from ppc (process_point_cloud) might call sys.exit on error.
    las_data = ppc.load_point_cloud(input_pc_path)

    print(f"Loading satellite image: {input_sat_path}")
    raster_dataset = ppc.load_satellite_image(input_sat_path)

    # In run_app.sh, --use_custom_transform is not specified, so custom_transform_func is None.
    # ppc.transform_coordinates (called by convert_point_cloud_to_image_coordinates)
    # will attempt to use pyproj by default.
    custom_transform_func = None

    print("Converting point cloud coordinates to image pixel coordinates...")
    pixel_x, pixel_y, valid_points = ppc.convert_point_cloud_to_image_coordinates(
        las_data,
        raster_dataset,
        custom_transform_func=custom_transform_func,
        x_offset=x_offset,
        y_offset=y_offset
    )
    # ppc.convert_point_cloud_to_image_coordinates already prints a warning if no points are valid.

    print("Extracting colors from satellite image...")
    colors = ppc.extract_colors_from_image(raster_dataset, pixel_x, pixel_y, valid_points)
    # ppc.extract_colors_from_image handles cases with no valid points by returning zeroed colors.

    print(f"Saving colorized point cloud to: {output_pc_path}")
    ppc.save_colorized_point_cloud(las_data, colors, output_pc_path)
    print("Colorization phase complete.")

    # --- 2. Visualization Phase ---
    print("\n--- Visualization Phase ---")

    # Check if the colorized output file exists, similar to run_app.sh
    if not os.path.exists(output_pc_path):
        print(f"Error: Colorized output file '{output_pc_path}' not found after colorization step.")
        print("Skipping visualization.")
        sys.exit(1) # Mimic behavior of run_app.sh check

    print(f"Loading colorized point cloud for visualization: {output_pc_path}")
    colorized_las_data = ppc.load_point_cloud(output_pc_path)

    print("Attempting to visualize colorized point cloud...")
    # ppc.visualize_with_pyvista itself checks if PyVista is available (ppc.VISUALIZATION_LIB)
    # and prints appropriate messages if it's not.
    if hasattr(ppc, 'visualize_with_pyvista'):
        ppc.visualize_with_pyvista(colorized_las_data, window_title="Colorized Point Cloud")
        if hasattr(ppc, 'VISUALIZATION_LIB') and ppc.VISUALIZATION_LIB == "pyvista":
            print("PyVista visualization window should be active.")
            print("Close the PyVista window to allow the script to terminate.")
        else:
            # This case implies visualize_with_pyvista exists but VISUALIZATION_LIB is not "pyvista"
            # which means PyVista import likely failed in process_point_cloud.py.
            # The visualize_with_pyvista function itself would have printed a message.
            print("Visualization was attempted, but PyVista may not be available.")
            print("Check console output from process_point_cloud.py's import attempts.")
    else:
        print("Error: 'visualize_with_pyvista' function not found in 'process_point_cloud.py'. Cannot visualize.")


    print("\nPhotogrammetry pipeline finished.")

if __name__ == "__main__":
    run_photogrammetry_pipeline()
