import laspy
import numpy as np
import open3d as o3d

# Load the LAZ file
laz_file = laspy.read("/workspaces/photogrammetry/data/colorized_point_cloud.laz")

# Extract points and colors
points = np.vstack(
    (laz_file.x, laz_file.z, laz_file.y)
).transpose()  # Swapped y and z to align elevation with Open3D's Y-up convention
colors = (
    np.vstack((laz_file.red, laz_file.green, laz_file.blue)).transpose() / 255.0
)  # Normalize colors to [0, 1]

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Create voxel grid from point cloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=2)

# Visualize (optional)
# o3d.visualization.draw_geometries([voxel_grid])

# Save the voxel grid (optional)
o3d.io.write_voxel_grid("data/house_vox.ply", voxel_grid)

print("Voxel grid created successfully.")
