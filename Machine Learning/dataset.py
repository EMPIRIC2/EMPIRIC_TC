import numpy as np
import h5py
import tensorflow as tf
import os
import glob

months = [1,2,3,4,11,12]

class hdf5_generator_v2:
    # similar to v1 but adding more data sources
    def __init__(self, file_paths, dataset="train", year_grouping_size=1):

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
                            yield (np.expand_dims(genesis[month], axis=-1), movement[months[month]]), np.sum(np.sum(output[i:i + self.year_grouping_size], axis=0)[:, month, :], -1)


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

def get_dataset(folder_path, batch_size=32, genesis_size=None, output_size=None, dataset="train", data_version=0):

    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))

    generator = None
    output_signature = None
    
    if data_version == 0:
        generator = hdf5_generator_v0
        genesis_size = (55, 105, 6)
        output_size = (110, 210, 6)
        output_signature = (
            tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
            tf.TensorSpec(shape=output_size, dtype=tf.float32)
        )

    if data_version == 1:
        generator = hdf5_generator_v1
        genesis_size = (55, 105,1)
        output_size = (542,)
        output_signature = (
            tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
            tf.TensorSpec(shape=output_size, dtype=tf.float32)
        )

    if data_version == 2:
        generator = hdf5_generator_v2
        genesis_size = (55, 105,1)
        movement_size = (13,)
        output_size = (542,)
        output_signature = ((
                tf.TensorSpec(shape=genesis_size, dtype=tf.float32),
                tf.TensorSpec(shape=movement_size, dtype=tf.float32)),
                tf.TensorSpec(shape=output_size, dtype=tf.float32)
            )

    dataset = tf.data.Dataset.from_generator(
            generator(file_paths, dataset=dataset),
            output_signature=output_signature
    )
    
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset

#test_v2 = get_dataset('../Training Data Generation/Data/v2/', data_version=1)
#for elem in test_v2.take(1):
    #print(elem)
