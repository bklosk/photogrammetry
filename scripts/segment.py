import open3d as o3d
import open3d.ml as _ml3d  # Common Open3D-ML utilities
import open3d.ml.torch as ml3d_torch  # PyTorch specific backend for Open3D-ML
import numpy as np
import laspy
import torch
import matplotlib.pyplot as plt


def load_point_cloud_from_laz(laz_file_path):
    """Loads point cloud data (points and colors) from a LAZ file."""
    try:
        inFile = laspy.read(laz_file_path)
    except Exception as e:
        print(f"Error reading LAZ file {laz_file_path}: {e}")
        return None, None

    points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

    if hasattr(inFile, "red") and hasattr(inFile, "green") and hasattr(inFile, "blue"):
        # Normalize colors. LAS colors are often 16-bit (0-65535).
        # Open3D and ML models typically expect colors in [0, 1].
        colors_r = np.array(inFile.red) / 65535.0
        colors_g = np.array(inFile.green) / 65535.0
        colors_b = np.array(inFile.blue) / 65535.0
        colors = np.vstack((colors_r, colors_g, colors_b)).transpose()
        # Ensure colors are float32 for PyTorch
        colors = colors.astype(np.float32)
    else:
        print("Color information not found in LAZ file. Using default gray.")
        colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)

    # Ensure points are float32 for PyTorch
    points = points.astype(np.float32)

    return points, colors


def main():
    laz_file = "../data/colorized_point_cloud.laz"  # Input file as per prompt # Adjusted path
    points_np, colors_np = load_point_cloud_from_laz(laz_file)

    if points_np is None:
        print("Failed to load point cloud. Exiting.")
        return

    print(f"Loaded {points_np.shape[0]} points from {laz_file}")

    # --- Semantic Segmentation with Open3D-ML ---
    model_name = "RandLANet"
    dataset_name = (
        "SemanticKITTI"  # Using SemanticKITTI pre-trained weights as an example
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # The pipeline automatically handles model download and setup for known models/datasets
        # Using the PyTorch specific pipeline from ml3d_torch
        pipeline = ml3d_torch.pipelines.SemanticSegmentation(
            model=model_name, dataset=dataset_name, device=device
        )
    except Exception as e:
        print(f"Error creating Open3D-ML pipeline: {e}")
        print(
            "Please ensure you have a working Open3D-ML setup and internet connection for model download."
        )
        print("This could be due to missing dependencies or issues with the model zoo.")
        print(f"Attempted to load model '{model_name}' with dataset '{dataset_name}'.")
        return

    # Prepare data for the model
    # The model expects a dictionary with 'points' and 'features'.
    # 'points' are XYZ coordinates. 'features' can be colors or other attributes.
    # The pre-trained RandLANet on SemanticKITTI might expect single-channel intensity features
    # or specific feature configurations. We'll try with colors first.

    data_for_inference = {
        "points": torch.from_numpy(points_np).to(device),
        "features": torch.from_numpy(colors_np).to(
            device
        ),  # Using RGB colors as features
    }

    print("Running inference with color features...")
    try:
        results = pipeline.run_inference(data_for_inference)
        print("Inference with color features succeeded.")
    except Exception as e_color_features:
        print(f"Error during inference with color features: {e_color_features}")
        print(
            "This might be due to a mismatch in expected feature dimensions (e.g., model expects intensity)."
        )
        print("Attempting fallback: using XYZ point coordinates as features...")
        try:
            data_xyz_features = {
                "points": torch.from_numpy(points_np).to(device),
                "features": torch.from_numpy(points_np).to(
                    device
                ),  # Using XYZ as features
            }
            results = pipeline.run_inference(data_xyz_features)
            print("Inference with XYZ point coordinates as features succeeded.")
        except Exception as e_xyz_features:
            print(
                f"Error during inference with XYZ point coordinates as features: {e_xyz_features}"
            )
            print("Failed to run inference with available feature sets. Exiting.")
            return

    pred_labels = results["predict_labels"].cpu().numpy()
    num_unique_labels = len(np.unique(pred_labels))
    print(f"Inference complete. Predicted {num_unique_labels} unique semantic classes.")

    # Visualize the segmented point cloud
    # Generate a colormap for the predicted labels.
    if num_unique_labels == 0:
        print("No labels predicted. Cannot visualize segmentation.")
        return

    # Use a qualitative colormap from matplotlib
    # Ensure we have enough colors for the predicted classes
    if num_unique_labels <= 10:  # tab10 has 10 distinct colors
        cmap = plt.get_cmap("tab10", num_unique_labels)
    elif num_unique_labels <= 20:  # tab20 has 20 distinct colors
        cmap = plt.get_cmap("tab20", num_unique_labels)
    else:
        # For more classes, 'viridis', 'plasma', etc., can be used, but they are sequential.
        # A custom map or hashing labels to colors might be better for many distinct classes.
        # Using 'viridis' as a fallback, normalized by the number of unique labels.
        cmap = plt.get_cmap("viridis", num_unique_labels)

    # Normalize labels to be 0 to N-1 for colormap indexing if they are not already
    unique_labels_arr = np.unique(pred_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels_arr)}
    indexed_labels = np.array([label_to_idx[label] for label in pred_labels])

    semantic_colors = cmap(indexed_labels / (num_unique_labels - 1e-9))[
        :, :3
    ]  # Get RGB, ignore alpha

    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(points_np)
    segmented_pcd.colors = o3d.utility.Vector3dVector(semantic_colors)

    print("Visualizing segmented point cloud. Close the window to exit.")
    o3d.visualization.draw_geometries(
        [segmented_pcd], window_name="Segmented Point Cloud (Open3D-ML)"
    )

    # For comparison, you can also visualize the original point cloud:
    # print("Visualizing original point cloud...")
    # original_pcd = o3d.geometry.PointCloud()
    # original_pcd.points = o3d.utility.Vector3dVector(points_np)
    # original_pcd.colors = o3d.utility.Vector3dVector(colors_np)
    # o3d.visualization.draw_geometries([original_pcd], window_name="Original Colorized Point Cloud")


if __name__ == "__main__":
    # Initial check to ensure Open3D-ML with PyTorch backend can be imported
    try:
        # The alias ml3d_torch was defined at the top of the file
        # Test basic import
        if ml3d_torch is not None and hasattr(ml3d_torch, "pipelines"):
            print("Open3D-ML with PyTorch backend imported successfully.")
        else:
            raise ImportError("ml3d_torch module not correctly initialized.")
    except ImportError as e:
        print(f"Error importing Open3D-ML PyTorch backend: {e}")
        print("Please ensure Open3D and a compatible version of PyTorch are installed.")
        print(
            "Follow the Open3D-ML installation instructions: https://www.open3d.org/docs/latest/getting_started.html#open3d-ml"
        )
        print("You might need to run: pip install open3d torch torchvision torchaudio")
        exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred while trying to import or use open3d.ml.torch: {e}"
        )
        print(
            "This could be due to an incomplete installation or environment issues (e.g., shared libraries)."
        )
        exit(1)

    main()
