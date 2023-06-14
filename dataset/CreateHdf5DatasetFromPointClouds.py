import open3d as o3d
import numpy as np
from CommonUtils3D.pyk4a_helpers import list_filenames
import os
import h5py
import argparse
import re


# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "tin_can",
    "04460130": "tower",
    "04468005": "train",
    "03085013": "keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02992529": "cellphone",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


def create_hdf5_dataset_from_point_clouds(
    ply_filenames_list, hdf5_output_file, n_points=2048
):
    ply_filenames = np.loadtxt(ply_filenames_list, dtype=str)
    print(len(ply_filenames))

    k = 0
    with h5py.File(hdf5_output_file, "w") as hdf5f:
        for synset_id, cate in synsetid_to_cate.items():
            ply_files_in_cate = [
                s for s in ply_filenames if re.match(r".*\/" + synset_id + "\/", s)
            ]
            hdf5f.create_dataset(cate, shape=(len(ply_files_in_cate), n_points, 3))
            for i, ply_file in enumerate(ply_files_in_cate):
                # synset_id = re.findall(r"\/([0-9]+)\/", ply_file)[0]
                # cate = synsetid_to_cate[synset_id]
                # Read .ply file
                pcd = o3d.io.read_point_cloud(ply_file)
                points = np.asarray(pcd.points)
                hdf5f[cate][i] = points
                #                print("hdf5f[cate].shape", hdf5f[cate].shape)
                # Write points to HDF5 file
                if k % 100 == 0:
                    print(f"Converted {k} meshes to point clouds")
                    print(f"Last mesh converted: {ply_file}")
                k += 1
    return


parser = argparse.ArgumentParser()
parser.add_argument("--ply_filenames_list", type=str, required=True)
parser.add_argument("--hdf5_output_file", type=str, required=True)
parser.add_argument("--n_points", type=int, default=2048)

## Example usage:
# python dataset/CreateHdf5DatasetFromPointClouds.py --ply_dataset_path /diskB/data/ShapeNetCore.v2/02691156 --hdf5_output_file /diskB/data/ShapeNetCore.v2/02691156.hdf5

if __name__ == "__main__":
    args = parser.parse_args()
    create_hdf5_dataset_from_point_clouds(
        args.ply_filenames_list, args.hdf5_output_file, args.n_points
    )
