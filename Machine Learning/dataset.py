import numpy as np
import h5py
import tensorflow as tf
import os
import glob

class hdf5_generator_v1:
    def __init__(self, file_paths, dataset="train", year_grouping_size=1, use_sites=False):

        self.file_paths = file_paths
        self.dataset = dataset
        self.year_grouping_size = year_grouping_size
        self.use_sites = use_sites

    def __call__(self):

        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                geneses = file[self.dataset + "_genesis"]
                movements = file[self.dataset + "_movement"]

                if self.use_sites:
                    outputs = file[self.dataset + "_sites"]
                else:
                    outputs = file[self.dataset + "_grids"]

                for genesis, movement, output in zip(geneses, movements, outputs):
                    if np.count_nonzero(genesis) != 0:  # data has been made
                        # switch the order of genesis matrix and divide output by number of years
                        for i in range(0,output.shape[0], self.year_grouping_size):
                            yield tf.transpose(genesis, [1,2,0]), np.sum(output[i:i + self.year_grouping_size], axis=0)

                    else:  # this sample was never generated
                        break



class hdf5_generator_v0:
    def __init__(self, file_paths, dataset="train"):

        self.file_paths = file_paths
        self.dataset = dataset

    def __call__(self):
        
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                for genesis, output in zip(file[self.dataset + "_genesis"], file[self.dataset + "_output"]):
                    if np.count_nonzero(genesis) != 0:# data has been made
                        # switch the order of genesis matrix and divide output by number of years
                        yield tf.transpose(genesis, [1, 2, 0]), np.flipud(output[:,:,[0,1,2,3,10,11]])
                    else: # this sample was never generated
                        break

def get_dataset(folder_path, batch_size=32, genesis_size=None, output_size=None, dataset="train", data_version=0):

    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))

    generator = None
    if data_version == 0:
        generator = hdf5_generator_v0
    if data_version == 1:
        generator = hdf5_generator_v1

    dataset = tf.data.Dataset.from_generator(
        generator(file_paths, dataset=dataset),
        output_types = (tf.float32, tf.float32),
        output_shapes = (genesis_size, output_size)
    )
    
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset


