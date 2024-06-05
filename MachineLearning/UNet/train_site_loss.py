import tensorflow as tf
from tensorflow import keras
from MachineLearning.dataset import get_dataset
from MachineLearning.UNet.unet import UNet
import time
import numpy as np
import pandas as pd
import math
import os

genesis_size_default = (55, 105, 1)
movement_size_default = (11, 13)
output_size_default = (110, 210, 1)

def getGridCell(lat, lon, resolution):
    '''
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    '''

    lat0, lat1, lon0, lon1 = -60, -5, 135, 240

    if lat < lat0 or lat >= lat1:
        raise "lat must be within the basin"

    if lon < lon0 or lon >= lon1:
        raise "lon must be within the basin"

    latCell = math.floor((lat - lat0) * 1/resolution)
    lonCell = math.floor((lon - lon0) * 1/resolution)

    return latCell, lonCell

def getHealthFacilityData():
        """
        :param file_paths:
        :return:
        """

        files = ['SPC_health_data_hub_Kiribati.csv', 'SPC_health_data_hub_Solomon_Islands.csv', 'SPC_health_data_hub_Tonga.csv', 'SPC_health_data_hub_Vanuatu.csv']
        file_paths = [os.path.join('../../HealthFacilities', file) for file in files]
        locations = []

        for file_path in file_paths:
            df = pd.read_csv(file_path)

            latitudes = df.loc[:, "LATITUDE: Latitude"]
            longitudes = df.loc[:, "LONGITUDE: Longitude"]


            ## adjust because our longitudes go from 0 to 360 not -180 to 180
            for i in range(len(longitudes)):
                if latitudes.loc[i] > -5 or latitudes.loc[i] < -60: continue # not in basin

                if longitudes.loc[i] < 0:
                    df.loc[i, "LONGITUDE: Longitude"] += 360

                locations.append((latitudes.loc[i], longitudes.loc[i]))

        return locations

sites = getHealthFacilityData()

def get_site_grid_weight():
    weights = np.ones(output_size_default)
    for site in sites:
        cell = getGridCell(*site, .5)
        weights[cell] += 1
    
    return weights

def Site_MSE(grid_weights):
    
    def site_mse(pred, true):

        return tf.reduce_sum(tf.square(grid_weights * (pred - true)))

def train_unet(data_folder, epochs=10, genesis_size=genesis_size_default, movement_size=movement_size_default, output_size=1):
    train_data = get_dataset(data_folder, data_version=3)
    test_data = get_dataset(data_folder, dataset="test", data_version=3)
    
    validation_data = get_dataset(data_folder, dataset="validation", data_version=3)
    
    model = UNet(genesis_size, movement_size, output_size)

    early_stopping = keras.callbacks.EarlyStopping(patience=5)
    checkpoint = keras.callbacks.ModelCheckpoint('models/unet_mean_{}.keras'.format(str(time.time())), save_best_only=True, save_weights_only=True, mode='min',
                                                 verbose=1)
    
    grid_weights = get_site_grid_weight()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=Site_MSE(grid_weights),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    model.fit(
        train_data,
        epochs=epochs,
        verbose=2,
        validation_data=validation_data,
        callbacks=[checkpoint, early_stopping]
    )

    model.evaluate(
        x=test_data,
    )

    model.save('models/unet_mean_{}.keras'.format(str(time.time())))

