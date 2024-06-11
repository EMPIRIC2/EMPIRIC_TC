from sklearn.neighbors import NearestNeighbors
from MachineLearning.dataset import get_dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt

class NearestNeighborPredictor:

    def __init__(self, data_dir):

        self.nearest_neighbors = None
        self.data_dir = data_dir
        output_dataset = get_dataset(data_dir, data_version=5).map(lambda x, y: y)
        self.output_data = np.empty(shape=(0,110, 210))

        for data in output_dataset.as_numpy_iterator():
            self.output_data = np.concatenate([self.output_data, data])

    def fit(self, save_path):
        train_data = get_dataset(self.data_dir, data_version=5).map(lambda x,y: x)

        data_array = np.empty(shape=(0,5775))
        for data in train_data.as_numpy_iterator():
            data_array = np.concatenate([data_array, np.reshape(data, newshape=(data.shape[0], -1))])

        nearest_neighbors = NearestNeighbors()
        nearest_neighbors.fit(data_array)

        pickle.dump(nearest_neighbors, open(save_path, 'wb'))
        self.nearest_neighbors = nearest_neighbors


    def load(self, save_path):

        nearest_neighbors = pickle.load(open(save_path, 'rb'))
        self.nearest_neighbors = nearest_neighbors
        return nearest_neighbors

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if self.nearest_neighbors is None: raise "No nearest neighbors' fit"

        indices = self.nearest_neighbors.kneighbors([x], return_distance=False)[0]

        return np.mean(self.output_data[indices], axis=0)
