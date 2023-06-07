import open3d as o3d

# Load the point cloud data from the .ply file on the remote server
point_cloud = o3d.io.read_point_cloud('./sandbox-data/bathtub_0001_256.ply')

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])