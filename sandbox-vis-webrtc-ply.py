import open3d as o3d
o3d.visualization.webrtc_server.enable_webrtc()
# Load the point cloud data from the .ply file on the remote server
point_cloud = o3d.io.read_point_cloud('./sandbox-data/bathtub_0003_256.ply')

# Visualize the point cloud
o3d.visualization.draw([point_cloud])