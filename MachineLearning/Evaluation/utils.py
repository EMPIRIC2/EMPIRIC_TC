import numpy as np
import math
from HealthFacilities.getHealthFacilityData import Sites

def get_site_values_from_grid(grids):
    """
    Gets the vectors of site values for many output grids
    :param: grids: the model outputs
    :returns: list of vectors of outputs for each site
    """

    site_outputs = []
    for output in grids:
        site_outputs.append(get_site_values(output))

    return site_outputs
def _get_outputs(dataset):
    return dataset.map(lambda x, y: y)

def _get_inputs(dataset):
    return dataset.map(lambda x,y: x)

def get_grid_cell(lat, lon, resolution):
    '''
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    '''

    lat0, lat1, lon0, lon1 = -60, -5, 135, 240

    if lat < lat0 or lat >= lat1:
        raise Exception("lat must be within the basin")

    if lon < lon0 or lon >= lon1:
        raise Exception("lon must be within the basin")

    latCell = math.floor((lat - lat0) * 1 / resolution)
    lonCell = math.floor((lon - lon0) * 1 / resolution)

    return latCell, lonCell

sites = Sites(1).sites

def get_site_values(grid):
    """
    Get the vector of values for each site from a grid output of a model

    returns: numpy array of output values at each site
    """
    site_values = np.zeros((len(sites),))

    for i, site in enumerate(sites):
        cell = get_grid_cell(*site, .5)
        site_values[i] = grid[cell]

    return site_values