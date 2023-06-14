# code grabbed from here: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?u=piojanu

import torch
import h5py


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_name):
        self.file_path = path
        self.dataset = None
        self.dataset_name = dataset_name
        with h5py.File(self.file_path, "r") as file:
            self.dataset_len = file[dataset_name].shape[0]

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.dataset_name]
        return self.dataset[index]

    def __len__(self):
        return self.dataset_len


# test
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--hdf5_file", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    args = parser.parse_args()
    dataset = H5Dataset(args.hdf5_file, args.dataset_name)
    print(len(dataset))
    print(dataset[0].shape)
    print(dataset[1].shape)
    print(dataset[2].shape)
    print(dataset[-1].shape)
    print(dataset[-1])
