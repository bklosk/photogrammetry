import laspy
import numpy as np
import open3d as o3d
from tqdm import tqdm

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
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=2)

# Estimate normals for Poisson reconstruction
print("Estimating normals...")
pcd.estimate_normals()

# Perform Poisson surface reconstruction
print("Performing Poisson surface reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# Remove low density vertices (optional cleanup)
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# The mesh from Poisson reconstruction doesn't have vertex colors
# We need to interpolate colors from the original point cloud
print("Interpolating colors to mesh vertices...")

# Get mesh vertices
mesh_vertices = np.asarray(mesh.vertices)
pcd_points = np.asarray(pcd.points)
pcd_colors = np.asarray(pcd.colors)

# For each mesh vertex, find the nearest point in the original point cloud
# and assign its color
from scipy.spatial import cKDTree

print("Building spatial tree for color interpolation...")
tree = cKDTree(pcd_points)

print("Finding nearest neighbors for color assignment...")
distances, indices = tree.query(mesh_vertices, k=1)

# Assign colors based on nearest neighbors
print("Assigning colors to mesh vertices...")
mesh_colors = pcd_colors[indices]
mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

# Verify colors were assigned
print(f"Mesh vertex colors assigned: {mesh.has_vertex_colors()}")
if mesh.has_vertex_colors():
    colors_check = np.asarray(mesh.vertex_colors)
    print(f"Color range: min={colors_check.min():.3f}, max={colors_check.max():.3f}")

# Visualize (optional)
# o3d.visualization.draw_geometries([mesh])

# Save the voxel grid and mesh
print("Saving files...")
o3d.io.write_voxel_grid("data/house_vox.ply", voxel_grid)

# Make sure to save with vertex colors enabled
success = o3d.io.write_triangle_mesh(
    "data/house_mesh.ply", mesh, write_vertex_colors=True
)
print(f"Mesh saved successfully: {success}")

print("Voxel grid and triangle mesh created successfully.")
print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
