from glob import glob
import h5py
import os
import numpy as np

def GetSampleCount(data_dir):
    file_paths = glob(os.path.join(data_dir, "AllData_*.hdf5"))
    datasets = ["train", "test", "validation"]
    counts = {"train": 0, "test": 0, "validation": 0}

    print(file_paths)
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as file:
            for dataset in datasets:
                geneses = file[dataset + "_genesis"]
                for genesis in geneses:
                    if np.count_nonzero(genesis) != 0: 
                        counts[dataset] = counts[dataset] + 1
                       
    print(counts)


if __name__ == "__main__":  
    GetSampleCount("/nesi/project/uoa03669/ewin313/storm_data/v4")