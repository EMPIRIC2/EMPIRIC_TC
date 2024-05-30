import numpy as np
import math
from HealthFacilities.getHealthFacilityData import Sites

def get_many_site_values(grids):
    """
    Gets the vectors of site values for many output grids
    :param: grids: the model outputs
    :returns: list of vectors of outputs for each site
    """
    site_outputs = []
    for output in grids:
        site_outputs.append(get_site_values(output))

    return np.array(site_outputs)

def get_outputs(dataset):
    """
    Takes a tensorflow dataset and returns a numpy array of the output data
    """
    outputs_ds =  dataset.map(lambda x, y: y)
    return np.squeeze(np.concatenate(list(outputs_ds.as_numpy_iterator()), axis=0))

def process_predictions(predictions):
    return np.squeeze(predictions)

def get_inputs(dataset):
    """
    Takes a tensorflow dataset and returns a tensorflow dataset with only the input data
    """
    return dataset.map(lambda x,y: x)

def get_grid_cell(lat, lon, resolution):
    """
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    """

    lat_min, lat_max = -60, -5
    lon_min, lon_max = 135, 240

    if not (lat_min <= lat < lat_max):
        raise Exception("lat must be within the basin")

    if not (lon_min <= lon < lon_max):
        raise Exception("lon must be within the basin")

    latCell = math.floor((lat - lat_min) * 1 / resolution)
    lonCell = math.floor((lon - lon_min) * 1 / resolution)

    return latCell, lonCell

sites = Sites(1)
def get_site_name(i):
    return sites.names[i]
def get_site_values(grid):
    """
    Get the vector of values for each site from a grid output of a model

    returns: numpy array of output values at each site
    """
    site_values = np.zeros((len(sites.sites),))

    for i, site in enumerate(sites.sites):
        cell = get_grid_cell(*site, .5)
        site_values[i] = grid[cell]

    return site_values