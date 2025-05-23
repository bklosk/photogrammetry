import open3d as o3d
import numpy as np
from landlab import RasterModelGrid
from landlab.components import OverlandFlow
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def create_topography_from_voxels(occ, vox_sz, min_xyz):
    """Extract surface topography from voxel occupancy grid"""
    nx, ny, nz = occ.shape

    # Create topography by finding the highest occupied voxel in each column
    # Note: Y is the vertical axis in our coordinate system after voxelization
    topography = np.zeros((nz, nx))  # Note: landlab uses (rows, cols) = (z, x)

    for k in range(nz):  # z is horizontal (north-south)
        for i in range(nx):  # x is horizontal (east-west)
            # Find highest occupied voxel in this column (along Y axis)
            column = occ[i, :, k]  # Extract column along Y (vertical) axis
            if np.any(column):
                highest_j = np.max(np.where(column)[0])
                # Convert voxel index to elevation (Y is vertical)
                topography[k, i] = min_xyz[1] + highest_j * vox_sz
            else:
                # Ground level if no voxels
                topography[k, i] = min_xyz[1]

    return topography


def simulate_flood(topography, spacing, rainfall_rate=0.001, duration=3600):
    """Run Landlab overlandflow simulation"""
    ny, nx = topography.shape

    # Create landlab grid
    grid = RasterModelGrid((ny, nx), xy_spacing=spacing)

    # Add topography to grid
    grid.add_field("topographic__elevation", topography.flatten(), at="node")

    # Initialize water depth
    grid.add_zeros("surface_water__depth", at="node")

    # Set boundary conditions (closed boundaries except maybe one edge for outflow)
    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)

    # Initialize overland flow component
    of = OverlandFlow(grid, steep_slopes=True)

    # Time stepping parameters
    dt = 1.0  # seconds
    total_time = 0.0

    print(f"Running flood simulation for {duration} seconds...")

    # Run simulation
    while total_time < duration:
        # Add rainfall
        grid.at_node["surface_water__depth"] += rainfall_rate * dt

        # Run one time step
        of.run_one_step(dt)

        total_time += dt

        if total_time % 300 == 0:  # Progress update every 5 minutes
            max_depth = np.max(grid.at_node["surface_water__depth"])
            print(f"Time: {total_time:.0f}s, Max water depth: {max_depth:.3f}m")

    # Get final water depths
    water_depths = grid.at_node["surface_water__depth"].reshape((ny, nx))

    return water_depths


def extrude_water_to_voxels(water_depths, topography, occ, vox_sz, min_xyz):
    """Convert 2D water depths back to 3D voxel representation"""
    nx, ny, nz = occ.shape
    water_voxels = np.zeros_like(occ, dtype=bool)

    # Note: topography is (nz, nx) corresponding to (z, x) grid
    # water_depths is (nz, nx) from the landlab simulation
    nz_topo, nx_topo = water_depths.shape

    for k in range(nz_topo):  # k corresponds to Z axis (north-south)
        for i in range(nx_topo):  # i corresponds to X axis (east-west)
            if water_depths[k, i] > 0.01:  # Minimum water depth threshold (1cm)
                # Ground elevation at this point
                ground_elev = topography[k, i]
                water_surface_elev = ground_elev + water_depths[k, i]

                # Convert elevations to voxel indices (Y is vertical axis)
                ground_j = int((ground_elev - min_xyz[1]) / vox_sz)
                water_j = int((water_surface_elev - min_xyz[1]) / vox_sz)

                # Find the actual surface (first empty voxel above ground)
                surface_j = ground_j
                for j in range(ground_j, ny):
                    if not occ[i, j, k]:  # Found first empty space along Y (vertical)
                        surface_j = j
                        break

                # Fill voxels with water from surface upward (along Y axis)
                for j in range(max(0, surface_j), min(ny, water_j + 1)):
                    if not occ[i, j, k]:  # Only fill empty space
                        water_voxels[i, j, k] = True

    return water_voxels


def create_colored_voxel_grid(occ, water_voxels, vox_sz, min_xyz):
    """Create colored voxel grid with buildings and water"""
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = vox_sz
    voxel_grid.origin = min_xyz

    # Colors: buildings = brown/gray, water = blue
    building_color = [0.6, 0.4, 0.2]  # Brown
    water_color = [0.2, 0.6, 1.0]  # Blue

    # Add building voxels
    for i in range(occ.shape[0]):
        for j in range(occ.shape[1]):
            for k in range(occ.shape[2]):
                if occ[i, j, k]:
                    voxel = o3d.geometry.Voxel()
                    voxel.grid_index = [i, j, k]
                    voxel.color = building_color
                    voxel_grid.add_voxel(voxel)

    # Add water voxels
    for i in range(water_voxels.shape[0]):
        for j in range(water_voxels.shape[1]):
            for k in range(water_voxels.shape[2]):
                if water_voxels[i, j, k]:
                    voxel = o3d.geometry.Voxel()
                    voxel.grid_index = [i, j, k]
                    voxel.color = water_color
                    voxel_grid.add_voxel(voxel)

    return voxel_grid


# Main execution
print("Loading voxel grid...")
vg = o3d.io.read_voxel_grid("data/house_vox.ply")  # your file
vox_sz = vg.voxel_size  # metres per voxel
min_xyz = vg.get_min_bound().astype(float)

# Allocate an empty Boolean occupancy array
nx, ny, nz = (np.array(vg.get_max_bound()) - min_xyz) / vox_sz
occ = np.zeros(tuple(np.ceil([nx, ny, nz]).astype(int)), dtype=bool)

# Mark solid voxels
print("Processing voxels...")
for v in vg.get_voxels():
    i, j, k = v.grid_index
    occ[i, j, k] = True

print(f"Voxel grid dimensions: {occ.shape}")
print(f"Voxel size: {vox_sz:.3f} meters")

# Extract topography
print("Extracting topography...")
topography = create_topography_from_voxels(occ, vox_sz, min_xyz)

# Run flood simulation
print("Starting flood simulation...")
water_depths = simulate_flood(
    topography, vox_sz, rainfall_rate=0.002, duration=1800
)  # 30 minutes

print(f"Maximum flood depth: {np.max(water_depths):.3f} meters")
print(f"Flooded area: {np.sum(water_depths > 0.01):.0f} grid cells")

# Convert water back to voxels
print("Converting water to voxels...")
water_voxels = extrude_water_to_voxels(water_depths, topography, occ, vox_sz, min_xyz)

# Create colored output
print("Creating colored voxel grid...")
colored_voxel_grid = create_colored_voxel_grid(occ, water_voxels, vox_sz, min_xyz)

# Export results
print("Exporting results...")
o3d.io.write_voxel_grid("data/flooded_house.ply", colored_voxel_grid)

# Save water depth map as image
plt.figure(figsize=(12, 8))
plt.imshow(water_depths, cmap="Blues", origin="lower")
plt.colorbar(label="Water Depth (m)")
plt.title("Flood Simulation Results")
plt.xlabel("Grid X")
plt.ylabel("Grid Y")
plt.savefig("data/flood_depth_map.png", dpi=300, bbox_inches="tight")
plt.close()

print("Flood simulation complete!")
print("Outputs:")
print("  - data/flooded_house.ply (colored voxel grid)")
print("  - data/flood_depth_map.png (water depth visualization)")

# Optional: visualize
print("Opening visualization...")
o3d.visualization.draw_geometries([colored_voxel_grid])
