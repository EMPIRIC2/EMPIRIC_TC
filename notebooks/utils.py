import glob
import numpy as np
import os
import h5py

def load_example_genesis_frequency(data_folder, sum_months = False):
    files = glob.glob(os.path.join(data_folder, "*.hdf5"))
    if len(files) < 1:
        return
        
    with h5py.File(files[0], 'r+') as file:
        genesis = file['train_genesis'][0]

        if sum_months:
            genesis = np.sum(genesis, axis=0)
        return genesis

def load_example_decadal_means(data_folder):
    files = glob.glob(os.path.join(data_folder, "*.hdf5"))
    if len(files) < 1:
        return
    with h5py.File(files[0], 'r+') as file:
        grid = file['train_grids'][0]
        grid = np.array(grid[-1])
        grid = np.sum(grid[:, :, :, 3:], axis=(-2, -1))
        return np.flipud(grid)
        
def move_last_two_months_first(matrix):
    return matrix[:,:,(4, 5, 0, 1, 2, 3),:]