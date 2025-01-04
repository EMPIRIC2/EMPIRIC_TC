from torch.utils import data
import torch
from MachineLearning.dataset import hdf5_generator_UNets
import numpy as np
import glob
import h5py
import os
import itertools


def normalize_input(data):
    max_val = np.max(data)
    min_val = np.min(data)
    return 2 * (data - min_val) / (max(max_val - min_val, 1e-3)) - 1


class IterDataset(data.IterableDataset):
    
    def __init__(
                self, 
                file_paths,
                dataset="train",
                min_category=0,
                max_category=1,
                n_samples=None,
                N_100_decades=10
            ):

        self.n_samples = n_samples
        self.file_paths = file_paths
        self.dataset = dataset
        self.indices = self._get_indices()
        self.min_category = min_category
        self.max_category = max_category
        self.N_100_decades = N_100_decades

    def _get_indices(self):
        indices = {}
        n_samples = 0
        for i,file_path in enumerate(self.file_paths):
            
            with h5py.File(file_path, "r") as file:
                
                geneses = file[self.dataset + "_genesis"]
                
                outputs = file[self.dataset + "_grids"][:,-1]
               
                if outputs.shape[0] != geneses.shape[0]: print("Wrong number of outputs!")
                    
                for j in range(geneses.shape[0]):
                    if (self.n_samples is not None and n_samples >= self.n_samples):
                        
                        break
                    if np.count_nonzero(geneses[j]) != 0 and np.count_nonzero(outputs[j]) != 0:  # data has been made
                        # switch the order of genesis matrix
                        # and divide output by number of years
                        indices.setdefault(i, []).append(j)
                        n_samples += 1
                        
                
        if self.n_samples is None:
            self.n_samples = n_samples
        return indices

    def __iter__(self):
        return iter(self.__getitem__())
            
    def __len__(self):
        return self.n_samples
    
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

        #padded_genesis = np.pad(upsampled_genesis, (lat_padding, lon_padding))

        # normalize and add channel dimension
        channel_dim = 0
        normalized_genesis = normalize_input(np.expand_dims(upsampled_genesis, axis=channel_dim))
        return normalized_genesis.astype(np.float32).copy()

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

        channel_axis = 0
        output_w_channels = np.expand_dims(mean_0_2_cat, axis=channel_axis).astype(np.float32)
        
        return output_w_channels.copy()

    def __getitem__(self):
        for i, file_path in enumerate(self.file_paths):
            if i not in self.indices.keys(): continue
                
            with h5py.File(file_path, "r") as file:
                 
                geneses = file[self.dataset + "_genesis"]
                
                outputs = file[self.dataset + "_grids"][:,self.N_100_decades-1]
                 
                for j in self.indices[i]:
                    yield {'x': self._preprocess_input(geneses[j]), 'y': self._preprocess_output(outputs[j])}

def get_pytorch_dataloader(
    folder_path,
    batch_size,
    dataset="train",
    n_samples=None,
    min_category=1,
    max_category=1,
    N_100_decades=10
):
    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))

    db = IterDataset(
        file_paths, 
        dataset=dataset, 
        n_samples=n_samples, 
        min_category=min_category, 
        max_category=max_category,
        N_100_decades=N_100_decades
    )

    dataloader = data.DataLoader(
        db,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return dataloader