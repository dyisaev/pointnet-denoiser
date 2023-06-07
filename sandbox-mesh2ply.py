import trimesh
import numpy as np

# Load the mesh from .off file
mesh = trimesh.load_mesh('/Users/dmitry/projects/3D-projects/data/ModelNet10/bathtub/train/bathtub_0001.off')

# Sample points uniformly from the surface of the mesh
point_cloud = trimesh.sample.sample_surface(mesh, count=2048) # 5000 points

# Save the point cloud to a .ply file
trimesh.points.PointCloud(point_cloud[0]).export('./sandbox-data/bathtub_0001_256.ply')