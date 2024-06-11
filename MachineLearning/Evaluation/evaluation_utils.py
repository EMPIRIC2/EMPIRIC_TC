import numpy as np
import math
from HealthFacilities.getHealthFacilityData import Sites

LAT_MIN, LAT_MAX = -60, -5
LON_MIN, LON_MAX = 135, 240


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

def process_predictions(predictions):
    return np.squeeze(predictions)

def get_grid_cell(lat: float, lon: float, resolution: float) -> tuple[int, int]:
    """
    Returns the grid cell that given latitude, longitude falls into for the specified resolution.
    The 0,0 cell is in the upper left corner, eg. latitude -5, longitude 135
    latCell index increases as latitude goes down and lonCell index increases as longitude goes up

    :return: indices of the lat and lon cells respectively
    """

    if not (LAT_MIN <= lat < LAT_MAX):
        raise Exception("lat must be within the basin")

    if not (LON_MIN <= lon < LON_MAX):
        raise Exception("lon must be within the basin")

    n_lat_cells = math.floor((lat_max - lat_min) * 1 / resolution)

    # flip the lat cell upside down
    latCell = n_lat_cells - math.floor((lat - lat_min) * 1 / resolution) - 1

    lonCell = math.floor((lon - lon_min) * 1 / resolution)

    return latCell, lonCell

def get_lat_lon_data_for_mesh(grid, resolution):

    # convert data into format for contour plot
    lats = [(grid.shape[0] - i) * resolution + LAT_MIN for i in range(grid.shape[0])]
    lons = [j * resolution + LON_MIN - 180 for j in range(grid.shape[1])]

    return lats, lons


sites = Sites(1)

def get_site_name(i):
    return sites.names[i]

def get_site_values(grid):
    """
    :param: grid: an array of values that is the model output on a latitude longitude grid.

    Returns a vector of values for each site
    values for each site are taken from the grid cell the site is located in

    returns: numpy array of output values at each site
    """
    site_values = np.zeros((len(sites.sites),))

    for i, site in enumerate(sites.sites):
        cell = get_grid_cell(*site, .5)
        site_values[i] = grid[cell]

    return site_values
