import glob
import os
import pickle
import time

import h5py
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from utils import get_random_year_combinations

months = [1, 2, 3, 4, 11, 12]

not_used_site_indices = [338, 409, 411, 418, 412, 419, 423, 426, 429, 500, 531, 511]

def normalize_input(data):
    max_val = np.max(data)
    min_val = np.min(data)
    return 2 * (data - min_val) / (max(max_val - min_val, 1e-3)) - 1

class hdf5_generator_unet_custom:
    def __init__(self, file_paths, dataset="train", n_samples=100, zero_inputs=False):
        self.file_paths = file_paths
        self.dataset = dataset
        self.n_samples = n_samples
        self.zero_inputs = zero_inputs

    def preprocess_input(self, genesis):
        # month sum
        genesis = np.sum(genesis, axis=0)

        # this is a simple way to do a nearest neighbor upsample
        upsampled_genesis = np.kron(genesis, np.ones((2, 2)))

        # pad. by upsampling first, the padding can be symmetric
        padded_genesis = np.pad(upsampled_genesis, ((1, 1), (7, 7)))

        # normalize and add channel dimension
        normalized_genesis = normalize_input(np.expand_dims(padded_genesis, axis=-1))
        return normalized_genesis

    def preprocess_output(self, output):
        mean_0_2_cat = np.flipud(np.sum(np.sum(output, axis=-1)[:, :, :3], axis=-1))
        output_w_channels = np.expand_dims(mean_0_2_cat, axis=-1)

        return output_w_channels

    def __call__(self):
        for file_path in self.file_paths:
            print(file_path)
            with h5py.File(file_path, "r") as file:
                geneses = file[self.dataset + "_genesis"]

                outputs = file[self.dataset + "_grids"]

                for genesis, output in zip(geneses, outputs):
                    if np.count_nonzero(genesis) != 0:  # data has been made
                        # switch the order of genesis matrix
                        # and divide output by number of years
                        if self.zero_inputs:
                            yield np.zeros((112, 224, 1)), self.preprocess_output(
                                output
                            )
                        else:
                            yield self.preprocess_input(
                                genesis
                            ), self.preprocess_output(output)

                    else:  # this sample was never generated
                        break

def get_dataset(
    folder_path,
    batch_size=32,
    dataset="train",
    data_version="unet-custom",
    n_samples=1,
):
    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))

    generator = None
    output_signature = None

    if data_version == "unet-custom":
        generator = hdf5_generator_unet_custom(file_paths, dataset=dataset, zero_inputs=False)
        genesis_size = (112, 224, 1)
        output_size = (110, 210, 1)
        output_signature = (
            tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
            tf.TensorSpec(shape=output_size, dtype=tf.float32),
        )

    start = time.time()
    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    )
    end = time.time()

    print("create dataset time: {}".format(end - start))

    # do not shuffle test dataset so we can get all outputs
    if dataset != "test":
        dataset = dataset.shuffle(100)
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset
