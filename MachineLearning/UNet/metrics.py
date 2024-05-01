
import numpy as np

from MachineLearning.dataset import get_dataset
from MachineLearning.UNet.unet import UNet
import os
import math
import pandas as pd


### Define metrics to evaluate the model
def getGridCell(lat, lon, resolution, basin):
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
        #__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        files = ['SPC_health_data_hub_Kiribati.csv', 'SPC_health_data_hub_Solomon_Islands.csv', 'SPC_health_data_hub_Tonga.csv', 'SPC_health_data_hub_Vanuatu.csv']
        file_paths = [os.path.join('./HealthFacilities', file) for file in files]
        locations = []

        for file_path in file_paths:
            df = pd.read_csv(file_path)

            latitudes = df.loc[:, "LATITUDE: Latitude"]
            longitudes = df.loc[:, "LONGITUDE: Longitude"]


            ## adjust because our longitudes go from 0 to 360 not -180 to 180
            for i in range(len(longitudes)):
                if latitudes.loc[i] > -5 or latitudes.loc[i] < -60: continue# not in basin

                if longitudes.loc[i] < 0:
                    df.loc[i, "LONGITUDE: Longitude"] += 360

                locations.append((latitudes.loc[i], longitudes.loc[i]))

        return locations

    
sites = getHealthFacilityData()

def get_site_values(grid):
    
    site_values = np.zeros((len(sites),))
    
    for i, site in enumerate(sites):
        cell = getGridCell(*site, .5, 'SP')
        site_values[i] = grid[cell]
        
    return site_values

def site_se(pred, true):
    
    squared_errors = []
    
    for site in sites:
        
        cell = getGridCell(*site, .5, 'SP')
        squared_error = (pred[cell] - true[cell])**2
        
        squared_errors.append(squared_error)
    return squared_errors

def site_mse(pred, true):
    return np.mean(squared_errors)


def _get_outputs():
    pass

def _get_inputs():
    pass

def get_mean_and_variance(data):
    return  np.mean(data, axis=0), np.var(predictions, axis=0)

def compute_test_site_statistics(outputs, predictions):
    
    site_outputs = []
    for output in outputs:
        site_outputs.append(get_site_values(output))
    
    site_predictions = []
    for prediction in predictions:
        site_predictions.append(get_site_values(prediction))
    
    
    get_mean_and_variance(site_outputs)
    get_mean_and_variance(site_predictions)
    
    
def compute_test_statistics(data_folder, model_path, sample_size,
    
    genesis_size_default = (55, 105, 1)
    movement_size_default = (11, 13)

    test_data = get_dataset(data_folder, data_version=3, dataset='test', batch_size=32)
    
    model = UNet(genesis_size_default, movement_size_default, 1)
    
    model.load_weights(model_path)
    
    outputs = _get_outputs(test_data)
    inputs = _get_inputs(test_data)
    
    predictions = model.predict(
            test_data,
            batch_size=32,
            verbose=2,
            steps=1
    )

    
def compute_test_grid_statistics(ouputs, predictions):

    ### get means and variances
    sample_mean, sample_variance = get_mean_and_variance(outputs)
    prediction_mean, prediction_variance = get_mean_and_variance(predictions)
       
    return sample_mean, sample_variance, prediction_mean, prediction_variance