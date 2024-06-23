import numpy as np
import h5py
import argparse
from os import path
import os
import glob

def countSamplesGenerated(file_path):
    count = 0
    with h5py.File(file_path, 'r') as file:
        for coeff in file['train_movement']:
            if np.count_nonzero(coeff) != 0:
                count += 1
        for coeff in file['test_movement']:
            if np.count_nonzero(coeff) != 0:
                count += 1
        for coeff in file['validation_movement']:
            if np.count_nonzero(coeff) != 0:
                count += 1
        
    print(file_path, count)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count the number of nonzero samples in hdf5 data file')
    parser.add_argument('file_path',  type=str, help="Path to hd5f file to count samples of")
    
    args = parser.parse_args()
    paths = None
    # check if file path is a folder
    if path.isdir(args.file_path):
       print("is dir")
       print(os.listdir(args.file_path)) 
       paths = glob.glob(os.path.join(args.file_path, "*.hdf5"))
    if path.isfile(args.file_path):
        paths = [args.file_path]
    if paths is not None:
        for p in paths:
            countSamplesGenerated(p)
