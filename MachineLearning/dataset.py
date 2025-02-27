import glob
import os
import random

import h5py
import numpy as np
import tensorflow as tf

months = [1, 2, 3, 4, 11, 12]

not_used_site_indices = [338, 409, 411, 418, 412, 419, 423, 426, 429, 500, 531, 511]

def normalize_input(data):
    max_val = np.max(data)
    min_val = np.min(data)
    return 2 * (data - min_val) / (max(max_val - min_val, 1e-3)) - 1

class hdf5_generator_nearest_neighbors:
    def __init__(
        self, 
        file_paths, 
        dataset="train", 
        min_category=3, 
        max_category=5,
        N_100_decades=10
    ):
        self.file_paths = file_paths
        self.dataset = dataset
        self.min_category = min_category
        self.max_category = max_category
        self.N_100_decades = N_100_decades

    @staticmethod
    def preprocess_input(genesis: np.ndarray):
        """

        @param genesis: (months = 6, lat = 55, lon = 105) np.ndarray
        @return: (5775,) np.ndarray
        """

        month_axis = 0
        genesis_month_sum = np.sum(genesis, axis=month_axis)
        flat_genesis = genesis_month_sum.flatten()
        return flat_genesis

    def _preprocess_output(self, output):

        month_axis = 2
        output_month_sum = np.sum(output, axis=month_axis)

        tc_category_axis = 2

        output_selected_categories = output_month_sum[
            :, :, self.min_category : self.max_category + 1
        ]
        output_category_sum = np.sum(output_selected_categories, axis=tc_category_axis)

        # the generated outputs are upside down from the genesis maps
        mean_0_2_cat = np.flipud(output_category_sum)

        return mean_0_2_cat

    def __call__(self):
        for file_path in self.file_paths:
            with h5py.File(file_path, "r") as file:
                geneses = file[self.dataset + "_genesis"]

                outputs = file[self.dataset + "_grids"][:, self.N_100_decades - 1]

                for genesis, output in zip(geneses, outputs):
                    if np.count_nonzero(genesis) != 0:  # data has been made
                        yield hdf5_generator_nearest_neighbors.preprocess_input(genesis), self._preprocess_output(
                            output
                        )

                    else:  # this sample was never generated
                        break


class hdf5_generator_UNets_Zero_Inputs:
    def __init__(
        self,
        file_paths,
        dataset="train",
        min_category=0,
        max_category=2,
    ):
        self.file_paths = file_paths
        self.dataset = dataset
        self.min_category = min_category
        self.max_category = max_category

    def _preprocess_output(self, output: np.ndarray):
        """
        This function preprocesses the outputs by summing over months
        and the specified TC categories
        @param output: (latitude = 110,
                        longitude = 210,
                        months = 6,
                        tc_categories = 6)
                shaped ndarray storing mean number of TCs passing over each
                cell over 10 years, separated by month and category.
        @return: a (latitude = 110, longitude = 210, channels = 1) shaped ndarray
            representing mean TCs passing over each cell per 10 years
        """

        month_axis = 2
        output_month_sum = np.sum(output, axis=month_axis)

        tc_category_axis = 2
        output_selected_categories = output_month_sum[
            :, :, self.min_category : self.max_category + 1
        ]
        output_category_sum = np.sum(output_selected_categories, axis=tc_category_axis)

        # the generated outputs are upside down from the genesis maps
        mean_0_2_cat = np.flipud(output_category_sum)

        channel_axis = -1
        output_w_channels = np.expand_dims(mean_0_2_cat, axis=channel_axis)

        return output_w_channels

    def __call__(self):
        for file_path in self.file_paths:

            with h5py.File(file_path, "r") as file:
                geneses = file[self.dataset + "_genesis"]

                outputs = file[self.dataset + "_grids"][:, -1]

                for genesis, output in zip(geneses, outputs):
                    if np.count_nonzero(genesis) != 0:  # data has been made
                        # switch the order of genesis matrix
                        # and divide output by number of years
                        yield np.zeros((112, 224, 1)), self._preprocess_output(output)

                    else:  # this sample was never generated
                        break


class hdf5_generator_UNets:
    def __init__(
        self,
        file_paths,
        min_category,
        max_category,
        dataset="train",
        N_100_decades = 10,
        crop_outputs=False
    ):
        self.file_paths = file_paths
        self.file_index = 0
        self.item_index = 0
        self.geneses = None
        self.outputs = None
        self.dataset = dataset
        self.min_category = min_category
        self.max_category = max_category
        self.N_100_decades = N_100_decades
        self.crop_outputs = crop_outputs

    def __iter__(self):
        return self
        
    def _preprocess_input(self, genesis: np.ndarray):
        """

        @param genesis: (months = 6, lat = 55, lon = 105) shaped np.ndarray
        @return: (lat = 110, lon=210, channels = 1) shaped np.ndarray
        with values normalized [-1, 1]
        """

        month_axis = 0
        genesis_month_sum = np.sum(genesis, axis=month_axis)

        # this is a simple way to do a nearest neighbor upsample
        scaling_factor = 2
        scaling_matrix = np.ones((scaling_factor, scaling_factor))
        upsampled_genesis = np.kron(genesis_month_sum, scaling_matrix)

        # we pad the inputs so that each dimension is divisible by 8
        # upsampled_genesis has shape (110, 210)
        # the closest shape with dimensions divisible by 8 is (112, 224)
        lat_padding = (1, 1)
        lon_padding = (7, 7)

        padded_genesis = np.pad(upsampled_genesis, (lat_padding, lon_padding))

        # normalize and add channel dimension
        normalized_genesis = normalize_input(np.expand_dims(padded_genesis, axis=-1))
        return normalized_genesis.copy()

    def _preprocess_output(self, output: np.ndarray):
        """
        This function preprocesses the outputs by summing over months
        and the specified TC categories
        @param output: (latitude = 110,
                        longitude = 210,
                        months = 6,
                        tc_categories = 6)
                shaped ndarray storing mean number of TCs passing over each
                cell over 10 years, separated by month and category.
        @return: a (latitude = 110, longitude = 210, channels = 1) shaped ndarray
            representing mean TCs passing over each cell per 10 years
        """
        
        month_axis = 2
        output_month_sum = np.sum(output, axis=month_axis)

        tc_category_axis = 2
        output_selected_categories = output_month_sum[
            :, :, self.min_category : self.max_category + 1
        ]
        output_category_sum = np.sum(output_selected_categories, axis=tc_category_axis)

        # the generated outputs are upside down from the genesis maps
        mean_cat = np.flipud(output_category_sum)
        lat_padding = (1, 1)
        lon_padding = (7, 7)
        
        if not self.crop_outputs:
            padded_mean = np.pad(mean_cat, (lat_padding, lon_padding))
        else:
            padded_mean = mean_cat
            
        channel_axis = -1
        output_w_channels = np.expand_dims(padded_mean, axis=channel_axis)
        return output_w_channels.copy()

    def __next__(self):
        
        if self.geneses is None or self.item_index >= len(self.geneses):
            # at the end of the file or haven't started, go to the next file
            if self.geneses is not None:
                self.file_index += 1
                
                # check that aren't at the end of the files
                if self.file_index >= len(self.file_paths): 
                    random.shuffle(self.file_paths)

            # update geneses and outputs to new file
            with h5py.File(self.file_paths[self.file_index], "r") as file:
                self.geneses = np.array(file[self.dataset + "_genesis"])

                self.outputs = np.array(file[self.dataset + "_grids"])[:,self.N_100_decades - 1]
            self.item_index = 0
            
        genesis = self._preprocess_input(self.geneses[self.item_index])
        
        output = self._preprocess_output(self.outputs[self.item_index])
        if np.count_nonzero(genesis) == 0 or np.count_nonzero(output) == 0:
            self.item_index += 1
            return self.__next__()
 
        self.item_index += 1
        return genesis, output
        

    def __call__(self):
        for file_path in self.file_paths:
            with h5py.File(file_path, "r") as file:
                geneses = file[self.dataset + "_genesis"]

                outputs = file[self.dataset + "_grids"][:, -1]

                for genesis, output in zip(geneses, outputs):
                    if np.count_nonzero(genesis) != 0:  # data has been made
                        # switch the order of genesis matrix
                        # and divide output by number of years

                        yield self._preprocess_input(genesis), self._preprocess_output(
                            output
                        )

                    else:  # this sample was never generated
                        break

            if file_path == self.file_paths[-1]:
                random.shuffle(self.file_paths)

def get_dataset(
    folder_path,
    batch_size=32,
    dataset="train",
    data_version="unet-custom",
    min_category=0,
    max_category=1,
    n_samples=1,
    N_100_decades = 10,
    crop_outputs=False
):
    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))

    generator = None
    output_signature = None
        
    if data_version == "unet":
        generator = hdf5_generator_UNets(
            file_paths,
            dataset=dataset,
            min_category=min_category,
            max_category=max_category,
            N_100_decades=N_100_decades,
            crop_outputs=crop_outputs
        )
        
        genesis_size = (112, 224, 1)
        if crop_outputs:
            output_size = (110, 210, 1)
        else:
            output_size = (112, 224, 1)
        output_signature = (
            tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
            tf.TensorSpec(shape=output_size, dtype=tf.float32),
        )
    elif data_version == "nearest_neighbors":
        generator = hdf5_generator_nearest_neighbors(
            file_paths,
            dataset=dataset,
            min_category=min_category,
            max_category=max_category,
            N_100_decades=N_100_decades
        )
        genesis_size = (5775,)
        output_size = (110, 210)
        output_signature = (
            tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
            tf.TensorSpec(shape=output_size, dtype=tf.float32),
        )
    elif data_version == "unet_zero_inputs":
        generator = hdf5_generator_UNets_Zero_Inputs(
            file_paths,
            dataset=dataset,
            min_category=min_category,
            max_category=max_category
        )

        genesis_size = (112, 224, 1)
        output_size = (112, 224, 1)
        output_signature = (
            tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
            tf.TensorSpec(shape=output_size, dtype=tf.float32),
        )
    else: raise Exception("Invalid data version")

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    )

    # do not shuffle test dataset so we can get all outputs
    if dataset != "test":
        dataset = dataset.shuffle(100)
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset
