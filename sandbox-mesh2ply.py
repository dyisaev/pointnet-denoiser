import trimesh
import numpy as np

# Load the mesh from .off file
mesh = trimesh.load_mesh('/diskB/data/ModelNet10_aligned/bathtub/train/bathtub_0003.off')

# Sample points uniformly from the surface of the mesh
point_cloud = trimesh.sample.sample_surface(mesh, count=2048) # 5000 points

# Save the point cloud to a .ply file
trimesh.points.PointCloud(point_cloud[0]).export('./sandbox-data/bathtub_0003_256.ply')