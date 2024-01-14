"""
    Take a list of storm tracks from storm and compute landfalls per month
"""
import numpy as np
import math
def BOUNDARIES_BASINS(idx):
    '''
    Copied from STORM model

    :param idx: basin index
    :return:
    '''
    if idx == 'EP':  # Eastern Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 180, 285
    if idx == 'NA':  # North Atlantic
        lat0, lat1, lon0, lon1 = 5, 60, 255, 359
    if idx == 'NI':  # North Indian
        lat0, lat1, lon0, lon1 = 5, 60, 30, 100
    if idx == 'SI':  # South Indian
        lat0, lat1, lon0, lon1 = -60, -5, 10, 135
    if idx == 'SP':  # South Pacific
        lat0, lat1, lon0, lon1 = -60, -5, 135, 240
    if idx == 'WP':  # Western Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 100, 180

    return lat0, lat1, lon0, lon1

def createMonthlyLandfallGrid(basin, resolution):
    '''
    Generate a grid to store TC monthly landfall statistics
    :param basin
    :param resolution: how many degrees each bin should be
    :return:
    '''
    # create lat lon grid (only needs to cover the basin) at 1 degree resolution
    # with [0] * 12 matrix in each

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    return np.zeros((
        math.ceil((lat1 - lat0) * 1/resolution),
        math.ceil((lon1 - lon0) * 1/resolution),
        12)
    )

def getGridCell(lat, lon, resolution, basin):
    '''
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    '''

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    latCell = math.floor((lat - lat0) * 1/resolution)
    lonCell = math.floor((lon - lon0) * 1/resolution)

    return latCell, lonCell

def averageLandfallsPerMonth(TC_data, basin, total_years, resolution):
    """
    Calculate average landfalls per month within lat, lon bins from synthetic TC data.
    For use as the training output of ML model.

    :param TC_data: list of synthetically generated TC data
    :param basin: the basin storms are being generated in
    :param total_years: number of years that synthetic data was generated over
    :param resolution: the lat, lon resolution to calculate statistics for in degrees
    :return: a lat, lon, month matrix for the basin with values
             the average number of landfalls in that month in the provided TC data
    """

    # TC_data is list of [year, storm_number, genesis_month, lat, lon, landfalllist]

    grid = createMonthlyLandfallGrid(basin, resolution)

    for i, storm in enumerate(TC_data):
        # for each track
        # create set of which grid cells it made landfall in
        landfallGrids = set()

        for timestep, val in enumerate(zip(storm[3], storm[4], storm[5])):

            lat, lon, landfall = val

            if landfall:
                landfallGrids.add(getGridCell(lat, lon, resolution, basin))


        for landfall in landfallGrids:
            latbin, lonbin = landfall
            grid[latbin][lonbin][storm[2]] += 1

    return grid

