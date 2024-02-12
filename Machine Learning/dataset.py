import numpy as np
import h5py
import tensorflow as tf
import os
import glob

class hdf5_generator:
    def __init__(self, file_paths, test=False):

        self.file_paths = file_paths
        self.dataset = "train"
        if test: self.dataset = "test"

    def __call__(self):
        print("Called generator")
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                for genesis, output in zip(file[self.dataset + "_genesis"], file[self.dataset + "_output"]):
                    if np.count_nonzero(genesis) != 0:# data has been made
                        print(genesis.shape)
                        # switch the order of genesis matrix and divide by number of years
                        yield tf.transpose(genesis, [1, 2, 0])/1000, output

                    else: # this sample was never generated
                        break

def get_dataset(folder_path, batch_size=32, genesis_size=None, output_size=None, test=False):

    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))
    print(file_paths)
    dataset = tf.data.Dataset.from_generator(
        hdf5_generator(file_paths, test=test),
        output_types = (tf.float32, tf.float32),
        output_shapes = (genesis_size, output_size)
    )
    print(dataset)
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset


