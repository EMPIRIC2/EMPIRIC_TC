import glob
import numpy as np
import h5py
import os

def compute_sample_means_for_file(file_path, dataset, categories=(0,1,2,3):
    
    with h5py.File(file_path, 'r+') as file:
        grids = file[dataset + "_grids"]
        
        try:
            grid_means = file.require_dataset(dataset + '_means', (len(grids), 110, 210), dtype='f') #(lat, lon, category)
        except Exception as e:
            print(e)
            # it was the wrong shape
            del file[dataset + '_means']
            grid_means = file.require_dataset(dataset + '_means', (len(grids), 110, 210), dtype='f') #(lat, lon, category)
            
        for i, grid in enumerate(grids):
            print(grid.shape)
            grid_means[i] = np.flipud(np.mean(np.sum(grid[:,:,:,:,:4], axis=(-1, -2)), axis=0))
        
def compute_sample_means(data_folder, dataset):
    file_paths = glob.glob(os.path.join(data_folder, "*.hdf5"))
    
    for file_path in file_paths:
        print("Computing sample means for: {}".format(file_path))
        compute_sample_means_for_file(file_path, dataset)
        
    