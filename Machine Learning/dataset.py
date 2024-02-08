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

        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                for genesis, movement in zip(file[self.dataset + "_genesis"], file[self.dataset + "_movement"]):
                    if np.count_nonzero(genesis) != 0 and np.count_nonzero(movement) != 0: # data has been made
                        yield genesis, movement

                    else: # this sample was never generated
                        break

def get_dataset(folder_path, batch_size=32, genesis_size=None, movement_size=None, test=False):

    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))

    dataset = tf.data.Dataset.from_generator(
        hdf5_generator(file_paths, test=test),
        output_types = (tf.float32, tf.float32),
        output_shapes = (genesis_size, movement_size)
    )

    batched_dataset = dataset.batch(batch_size)

    return batched_dataset


