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
    np.vstack((laz_file.red, laz_file.green, laz_file.blue)).transpose() / 65535.0
)  # Normalize colors from 16-bit (0-65535) to [0, 1]

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Create voxel grid from point cloud
print("Creating voxel grid...")
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=2)

# Save the voxel grid
print("Saving voxel grid...")
o3d.io.write_voxel_grid("/workspaces/photogrammetry/data/house_vox.ply", voxel_grid)

print("Voxel grid created and saved successfully.")
