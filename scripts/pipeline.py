#!/usr/bin/env python3
"""
Photogrammetry Pipeline Script

This script orchestrates the point cloud processing workflow.
Currently, it focuses on colorizing a point cloud using satellite imagery.
"""

import os
import subprocess


def run_command(command):
    """Helper function to run a shell command and print its output."""
    print(f"Executing: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"Error: Command failed with exit code {process.returncode}")
            return False
    except Exception as e:
        print(f"An error occurred while trying to run the command: {e}")
        return False
    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)  # Get /workspaces/photogrammetry

    # Hardcoded parameters
    input_pc_path = os.path.join(workspace_root, "data/point_cloud.laz")
    input_sat_path = os.path.join(workspace_root, "data/boulder_flyover/images/n2w235.tif")
    output_pc_path = os.path.join(workspace_root, "data/colorized_point_cloud.laz")
    x_offset = "0.0"
    y_offset = "0.0"
    use_custom_transform = False

    process_script_path = os.path.join(script_dir, "process_point_cloud.py")

    # --- 1. Colorize Point Cloud ---
    print("\n--- Starting Point Cloud Colorization ---")
    colorize_command = [
        "python",
        process_script_path,
        "--input_pc",
        input_pc_path,  # Use hardcoded path
        "--input_sat",
        input_sat_path,  # Use hardcoded path
        "--output_pc",
        output_pc_path,  # Use hardcoded path
        "--x_offset",
        x_offset,  # Use hardcoded value
        "--y_offset",
        y_offset,  # Use hardcoded value
    ]
    if use_custom_transform:  # Use hardcoded value
        colorize_command.append("--use_custom_transform")

    if not run_command(colorize_command):
        print("Colorization failed. Exiting pipeline.")
        return

    print("\n--- Point Cloud Colorization Completed ---")
    print(f"Colorized point cloud saved to: {output_pc_path}")  # Use hardcoded path

    # Visualization phase and related calls have been removed.

    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()
