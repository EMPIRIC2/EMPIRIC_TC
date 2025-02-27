from saved_models import ddpm_unet
from MachineLearning.NearestNeighbors.nearest_neighbors import NearestNeighborsRegressor
import os
import numpy as np
from MachineLearning.dataset import normalize_input
from STORM.storm import STORM
from neuralop.models import FNO2d
import torch

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class FNO_model:
    

    def __init__(self, hundred_decades=False):
        self.model = FNO_model.load_model(hundred_decades)

    def predict(self, dataloader):
        prediction_list = []
        for i, batch in enumerate(dataloader):
            pred = self.model(batch['x'])
            prediction_list.append(pred.detach().numpy())
        return np.concatenate(prediction_list, axis=0)

    @staticmethod
    def load_model(hundred_decades=False):
        
        MODEL_CONFIG = {
            "n_modes_height": 16, 
            "n_modes_width": 16,
            "in_channels": 1,
            "hidden_channels": 32,
            "projection_channels": 64,
            "factorization": 'tucker', 
            "rank": 0.42
        }

        if hundred_decades:
            model_path = os.path.join(__location__, "FNO_1733842125.6365902.keras")

        else:
            model_path = os.path.join(__location__, "FNO_1729471215.0322773.keras")

        model = FNO2d(
            MODEL_CONFIG.pop("n_modes_height"),
            MODEL_CONFIG.pop("n_modes_width"),          
            **MODEL_CONFIG
        )

        model.load_state_dict(torch.load(model_path, weights_only=True))

        return model
    
    @staticmethod
    def preprocess_input(genesis: np.ndarray):
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

    def __call__(self, x):
        return self.model(x)[0].detach().numpy()

class DDPMUNet_model:
    
    MODEL_CONFIG = {
        "img_size": (112, 224, 1),
        "output_size": (112, 224, 1),
        "has_attention": [False, False, True],
        "interpolation": "bilinear",
        "widths": [16, 32, 64],
        "include_temb": False
    }

    def __init__(self, hundred_decades=False):
        self.model = DDPMUNet_model.load_model(hundred_decades)

    @staticmethod
    def load_model(hundred_decades):
        if not hundred_decades:
            model_path = os.path.join(__location__, "DDPM-Unet_1733500332.331749.keras")
        else:
            model_path = os.path.join(__location__, "DDPM-Unet_1733766375.4331155.keras")

        model = ddpm_unet.build_model(**DDPMUNet_model.MODEL_CONFIG)
        model.load_weights(model_path)
        return model

    @staticmethod
    def preprocess_input(genesis: np.ndarray):
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

    def __call__(self, x):
        x = self.preprocess_input(x)
        x = x[np.newaxis, :]

        # crop the UNet output so it removes the padded edges
        lat_crop = (1, 1)
        lon_crop = (7, 7)
        x = x[lat_crop[0]: x.shape[0] - (lat_crop[1] - 1), lon_crop[0] : x.shape[1] - (lon_crop[1] - 1)]
        
        return self.model(x)[0]

    def predict(self, dataset):
        predictions = self.model(dataset)
        # crop the UNet output so it removes the padded edges
        lat_crop = (1, 1)
        lon_crop = (7, 7)
        print(predictions.shape)
        predictions = predictions[:, lat_crop[0]: predictions.shape[1] - (lat_crop[1]), lon_crop[0] : predictions.shape[2] - (lon_crop[1]), :]
        return predictions
        

class NearestNeighbors_model:

    model_path = os.path.join(__location__, "nearest_neighbors_cat_3_5.pkl")

    def __init__(self, train_data_dir):
        self.model = NearestNeighbors_model.load_model(train_data_dir)

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

    @staticmethod
    def load_model(train_data_dir):
        nearest_neighbors_regressor = NearestNeighborsRegressor(train_data_dir, min_category=3, max_category=5)

        nearest_neighbors_regressor.load(NearestNeighbors_model.model_path)

        return nearest_neighbors_regressor

    def __call__(self, x):

        x = NearestNeighbors_model.preprocess_input(x)

        return self.model(x)

    def predict(self, x):
        return self.model(x)
        
class STORM_model:
    def __init__(self, *args, **kwargs):
        self.model = STORM(*args, **kwargs)

    def __call__(self, x):
        return self.model(x)

