import pickle

import numpy as np
from sklearn.neighbors import NearestNeighbors

from MachineLearning.dataset import get_dataset

class NearestNeighborsRegressor:
    """
    This class implements a regression model that
    applies a nearest neighbor algorithm to find
    the 5 closest training example inputs and then
    returns the mean of their outputs as a prediction
    """

    def __init__(self, data_dir, min_category=0, max_category=1):
        """
        @param data_dir: directory for the training data
        """
        self.nearest_neighbors = None
        self.data_dir = data_dir
        self.min_category = min_category
        self.max_category = max_category
        output_dataset = get_dataset(data_dir, data_version="nearest_neighbors", min_category=min_category, max_category=max_category).map(lambda x, y: y)
        self.output_data = np.empty(shape=(0, 110, 210))

        for data in output_dataset.as_numpy_iterator():
            self.output_data = np.concatenate([self.output_data, data])

    def fit(self, save_path):
        """
        Fits the nearest neighbor algorithm on the training data in self.data_dir
        and saves it to save_dir in pickle format

        @param save_path: file path to save the nearest neighbors
        """

        train_data = get_dataset(self.data_dir, data_version="nearest_neighbors", min_category=self.min_category, max_category=self.max_category).map(lambda x, y: x)

        data_array = np.empty(shape=(0, 5775))
        for data in train_data.as_numpy_iterator():
            data_array = np.concatenate([data_array, data])

        nearest_neighbors = NearestNeighbors()
        nearest_neighbors.fit(data_array)

        pickle.dump(nearest_neighbors, open(save_path, "wb"))
        self.nearest_neighbors = nearest_neighbors

    def load(self, save_path):
        """
        Loads a saved nearest neigbhors algorithm.
        Assumes that it is already fitted.
        @param save_path: the path to the save file (in pickle format)
        @return: nearest neighbor object
        """
        nearest_neighbors = pickle.load(open(save_path, "rb"))
        self.nearest_neighbors = nearest_neighbors
        return nearest_neighbors

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x: np.ndarray):
        """
        Predicts an output by first finding the 5 nearest neighbors to each
        example in x and then returning the mean of their outputs respectively.
        @param x: a 1d or 2d ndarray, either a single (5775,) input to predict
        or a batch (None, 5775) of inputs to predict on
        @return: a (None, 110, 210) size ndarray
        """
        if self.nearest_neighbors is None:
            raise "Nearest neighbors' not fitted"

        if len(x.shape) == 1:
            assert x.shape == (5775,)
            x = [x]
        elif len(x.shape) == 2:
            assert x.shape[1] == 5775

        indices = self.nearest_neighbors.kneighbors(x, return_distance=False)
        print(len(x))
        return np.mean(self.output_data[indices], axis=1)
