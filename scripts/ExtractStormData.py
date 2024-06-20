import h5py
import glob
import numpy as np

def extractStormDataForSite(file_path):
    with h5py.File(file_path, 'r') as file:
        outputs = file["train_sites"]
        dist_indices = np.sort(np.argpartition(np.sum(np.sum(np.sum(outputs[0], axis=-1), axis=-1), axis=0), -10)[-10:])
        print(dist_indices)
        sample = outputs[0,:,dist_indices,:,:]
        print(np.sum(np.sum(np.sum(np.sum(outputs[0], axis=-1), axis=-1), axis=0)))
        np.save('extracted_data_for_10_sites.npy', sample)
        np.save('sites_extracted.npy', dist_indices)
    
file_paths = glob.glob('/nesi/project/uoa03669/ewin313/storm_data/v2/*.hdf5')
extractStormDataForSite(file_paths[0])
