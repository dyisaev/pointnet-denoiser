import open3d as o3d
import numpy as np
from CommonUtils3D.ConvertMeshToPointCloud import convert_mesh_to_pointcloud
from CommonUtils3D.pyk4a_helpers import read_obj,list_filenames

import os
import argparse

def convert_shapenet_meshes_to_point_clouds(dataset_folder, number_of_points, sample_type='uniform', init_factor=5):
    """
    Convert all meshes in the dataset_folder to point clouds and save them in the output_folder.
    """
    filenames = list_filenames(dataset_folder)
    print(len(filenames))
    successful_conversions = []
    unsuccessful_conversions = []
    for fname in filenames:
        #print('fname:',fname)
        try:
            mesh = read_obj(fname+'.obj')
            #mesh = o3d.io.read_triangle_mesh(fname+'.obj')
            #print(mesh)
            #print ('read mesh')
        except Exception as e:
            print(f'Could not read mesh {fname}')
            #print(e)
            unsuccessful_conversions.append(fname)
            continue
        try:
            point_cloud = convert_mesh_to_pointcloud(mesh, number_of_points, sample_type=sample_type,\
                                                     compute_normals= False if sample_type=='uniform' else True, init_factor=init_factor)
            #print ('converted mesh to point cloud')
        except:
            print(f'Could not convert mesh {fname} to point cloud')
            unsuccessful_conversions.append(fname)
            continue
        try:
            o3d.io.write_point_cloud(f'{fname}_{sample_type}_{number_of_points}.ply', point_cloud)
            #print ('wrote point cloud')
        except:
            print(f'Could not write point cloud {fname}')
            unsuccessful_conversions.append(fname)
            continue
        successful_conversions.append(f'{fname}_{sample_type}_{number_of_points}.ply')
        #print(len(successful_conversions))
        if len(successful_conversions) % 100 == 0:
            print(f'Converted {len(successful_conversions)} meshes to point clouds')
            print(f'Last mesh converted: {fname}')
    return successful_conversions,unsuccessful_conversions

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--number_of_points', type=int, required=True)
parser.add_argument('--sample_type', type=str, required=True)
parser.add_argument('--init_factor', type=int, required=False, default=5)
parser.add_argument('--logfile', type=str, required=False, default='convert_shapenet_meshes_to_ply.log')
## Example usage:
# python dataset/convert_shapenet_meshes_to_ply.py --dataset_path /diskB/data//ShapeNetCore.v2 --number_of_points 2048 --sample_type uniform --init_factor 5 --logfile convert_shapenet_meshes_to_ply

## Example usage - uniform 2048:
# nohup python dataset/convert_shapenet_meshes_to_ply.py --dataset_path /diskB/data/ShapeNetCore.v2 --number_of_points 2048 --sample_type uniform --init_factor 5 --logfile /diskB/data/convert_shapenet_meshes_to_ply_20230613 > sandbox-data/conv_log.txt &

if __name__ == '__main__':
    args=parser.parse_args()
    dataset_path=args.dataset_path
    number_of_points=args.number_of_points
    sample_type=args.sample_type
    init_factor=args.init_factor
    logfile=args.logfile

    succesful_conversions,unsuccessful_conversions =\
            convert_shapenet_meshes_to_point_clouds(dataset_path, number_of_points,sample_type=sample_type, init_factor=init_factor)
   
    log_successful = f'{args.logfile}_successful.log'
    log_unsuccessful = f'{args.logfile}_unsuccessful.log'
    with open(log_successful, 'w') as f:
        for item in succesful_conversions:
            f.write("%s\n" % item)
    with open(log_unsuccessful, 'w') as f:
        for item in unsuccessful_conversions:
            f.write("%s\n" % item)