import numpy as np
import h5py
import tensorflow as tf
import os
import glob
import itertools
import random
from utils import *
from sklearn.decomposition import PCA
import pickle
import time
print("loaded packages")
months = [1,2,3,4,11,12]

not_used_site_indices = [338, 409, 411, 418, 412, 419, 423, 426, 429, 500, 531, 511]

def normalize_input(data):
    max_val = np.max(data)
    min_val = np.min(data)
    return 2*(data - min_val)/(max(max_val-min_val, 1e-3)) - 1

def computePCADecompForGeneratorV2(file_paths, dataset, save_path, n_samples = 100, n_components=4):
    print("Computing PCA")
    pca_genesis = PCA(n_components=n_components)
    pca_movement = PCA(n_components=n_components)
    month = 3
    def genesis_generator():
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as file:
                geneses = file[dataset + "_genesis"]
                movements = file[dataset + "_movement"]
                for genesis, movement in zip(geneses, movements):
                    if np.count_nonzero(movement) != 0:  # data has been made
                        yield genesis[month]
    
    def movement_generator():
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as file:
                movements = file[dataset + "_movement"]
                
                for movement in movements:
                    if np.count_nonzero(movement) != 0:  # data has been made
                        yield movement
    
    geneses = [np.array(genesis).flatten() for i,genesis in enumerate(genesis_generator()) if i < n_samples]
    pca_genesis.fit(geneses)
    del geneses
    
    movements = [np.array(movement).flatten() for i,movement in enumerate(movement_generator()) if i < n_samples]
    pca_movement.fit(movements)
    del movements
 
    pickle.dump(pca_movement, open(save_path + "pca_movement.pkl", "wb"))
    pickle.dump(pca_genesis, open(save_path + "pca_genesis.pkl", "wb"))


    
class hdf5_generator_v3:
    def __init__(self, file_paths, dataset="train", month=3):

        self.file_paths = file_paths
        self.dataset = dataset
        self.month = month

    def __call__(self):

        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                geneses = file[self.dataset + "_genesis"]
                movements = file[self.dataset + "_movement"]

                outputs = file[self.dataset + "_grids"]

                for genesis, movement, output in zip(geneses, movements, outputs):
                    if np.count_nonzero(movement) != 0:  # data has been made
                        # switch the order of genesis matrix and divide output by number of years

                        yield (np.expand_dims(genesis[self.month], axis=-1), movement), np.expand_dims(np.sum(output[:,self.month,:4], axis=-1), -1)

                    else:  # this sample was never generated
                        break

class hdf5_generator_v2:
    # similar to v1 but adding more data sources
    def __init__(self, file_paths, pca_path, dataset="train", year_grouping_size=100, zero_inputs=False, n_samples=None):

        self.file_paths = file_paths
        self.dataset = dataset
        self.pca_genesis = pickle.load(open(pca_path + "pca_genesis.pkl", 'rb'))
        self.pca_movement = pickle.load(open(pca_path + "pca_movement.pkl", 'rb'))
        self.year_grouping_size = year_grouping_size
        self.zero_inputs = zero_inputs
        self.n_samples = n_samples
        
        for file_path in file_paths: self.process_data(file_path)
    
    def create_histogram(self, output, year_indices, month=0):
        hist = np.zeros((542, 8))
        print("Creating histogram")
        # make a histogram of values
        for year_index in year_indices:
            summed_counts = np.sum(np.sum(output[list(year_index)], axis=0)[:, month, :], -1)
            summed_counts = np.clip(summed_counts, 0, 7)
            # now add summed counts to the histogram
            for i in range(542):
                
                hist[i][int(summed_counts[i])] = hist[i][int(summed_counts[i])] + 1
                
            # remove the duplicate sites from the histogram
        indices = [i for i in range(542) if i not in not_used_site_indices]
        hist = hist[indices]
            
        density = hist/1000
            
        return density
    
    def process_data(self, file_path):
        month=0
        with h5py.File(file_path, 'r+') as file:

            geneses = file[self.dataset + "_genesis"]
            movements = file[self.dataset + "_movement"]
            
            outputs = file[self.dataset + "_sites"]
            
            try:
                histograms = file.require_dataset(self.dataset + '_histograms', (len(geneses), 530, 8), dtype='f') #(months, sites, count)
            except Exception as e:
                print(e)
                # it was the wrong shape
                del file[self.dataset + '_histograms']
                histograms = file.require_dataset(self.dataset + '_histograms', (len(geneses), 530, 8), dtype='f') #(months, sites, count)

            geneses_pca = file.require_dataset(self.dataset + '_genesis_pca', (len(geneses), 4), dtype='f')
            movements_pca = file.require_dataset(self.dataset + '_movement_pca', (len(geneses), 4), dtype='f')
            
            #if np.count_nonzero(histograms[0]) != 0: return # data has already been processed
            print(histograms[0])
            print(geneses_pca[0])
            #randomly select combinations of years for training
            year_indices = get_random_year_combinations(1000, 1000, self.year_grouping_size)

            for j, (genesis, movement, output) in enumerate(zip(geneses, movements, outputs)):
                print(j)
                if np.count_nonzero(genesis) != 0:  # data has been made
                # switch the order of genesis matrix and divide output by number of years
                    #if np.count_nonzero(movements_pca) == 0: # data has not been made, convert data
                        
                        histogram = self.create_histogram(output, year_indices, month=month)
                        histograms[j] = histogram

                        geneses_pca[j] = self.pca_genesis.transform(np.array(genesis[month]).reshape(1, -1))[0]
                        movements_pca[j] = self.pca_movement.transform(np.array(movement).reshape(1, -1))[0]

    def __call__(self):
        #samples_generated = 0
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                geneses_pca = file[self.dataset + "_genesis_pca"]
                movements_pca = file[self.dataset + "_movement_pca"]

                hists = file[self.dataset + "_histograms"]
                
                #randomly select combinations of years for training
                year_indices = get_random_year_combinations(1000, 1000, self.year_grouping_size)
                
                for genesis, movement, hist in zip(geneses_pca, movements_pca, hists):
                    if np.count_nonzero(genesis) != 0:  # data has been made
                            
                        if self.zero_inputs:
                            yield (np.zeros(4,), np.zeros(4,)), hist
                        else:
                            yield (genesis, movement), hist
                            
                    else:  # this sample was never generated
                        break
                        
class hdf5_generator_v1:
    def __init__(self, file_paths, dataset="train", year_grouping_size=30):

        self.file_paths = file_paths
        self.dataset = dataset
        self.year_grouping_size = year_grouping_size

    def __call__(self):
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                geneses = file[self.dataset + "_genesis"]
                movements = file[self.dataset + "_movement"]

                outputs = file[self.dataset + "_sites"]

                for genesis, movement, output in zip(geneses, movements, outputs):
                    if np.count_nonzero(genesis) != 0:  # data has been made
                        # switch the order of genesis matrix and divide output by number of years
                       
                        for i in range(0,output.shape[0], self.year_grouping_size):
                            month = 3
                            yield np.expand_dims(genesis[month], axis=-1), np.sum(np.sum(output[i:i + self.year_grouping_size], axis=0)[:, month, :], -1)

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

def get_dataset(folder_path, pca_path, batch_size=32, dataset="train", data_version=0, n_samples=1, generate_pcas=False, zero_inputs=False):
    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))

    generator = None
    output_signature = None
    
    if data_version == 0:
        generator = hdf5_generator_v0(file_paths, dataset=dataset)
        genesis_size = (55, 105, 6)
        output_size = (110, 210, 6)
        output_signature = (
            tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
            tf.TensorSpec(shape=output_size, dtype=tf.float32)
        )
        
    if data_version == 1:
        generator = hdf5_generator_v1(file_paths, dataset=dataset)
        genesis_size = (55, 105,1)
        output_size = (542,)
        output_signature = (
            tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
            tf.TensorSpec(shape=output_size, dtype=tf.float32)
        )

    if data_version == 2:
        
        if generate_pcas:
            computePCADecompForGeneratorV2(file_paths, dataset, pca_path)
        
        generator = hdf5_generator_v2(file_paths, pca_path, dataset=dataset, n_samples=n_samples, zero_inputs=zero_inputs)
        genesis_size = (4,)
        movement_size = (4,)
        output_size = (542,8)
        output_signature = ((
                tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
                tf.TensorSpec(shape=movement_size, dtype=tf.float32)),
                tf.TensorSpec(shape=output_size, dtype=tf.float32)
            )
    if data_version == 3:
        generator = hdf5_generator_v3(file_paths, dataset=dataset)
        genesis_size = (55, 105, 1)
        movement_size = (13,11)
        output_size = (110, 210, 11)
        output_signature = ((
                                tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
                                tf.TensorSpec(shape=movement_size, dtype=tf.float32)),
                            tf.TensorSpec(shape=output_size, dtype=tf.float32)
        )
    start = time.time()
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    end = time.time()
    
    print("create dataset time: {}".format(end-start))
          
    # do not shuffle test dataset so we can get all outputs
    if dataset != "test":
        dataset = dataset.shuffle(100)
    batched_dataset = dataset.take(100).batch(batch_size)

    return batched_dataset

'''
#test_v2 = get_dataset('../storm_data/v2/', data_version=2)
for elem in test_v2.take(1):
    (gen_matrix, _), output = elem
    print(np.max(gen_matrix))
    print(np.mean(gen_matrix))
    print(output)
   #print(np.median(gen_matrix))
   
'''
