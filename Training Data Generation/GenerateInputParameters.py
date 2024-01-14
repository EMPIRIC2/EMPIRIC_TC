"""
    Generate randomized input parameters for the STORM model that can be
    used to create a training dataset for machine learning.

    For Now we only care about TC genesis and movement, so we need to generate inputs for
    # storms per year (should be fairly constant)
    genesis month (also assume constant)
    Genesis location (randomize weighted based on prob. per 1 degree x 1 degree boc)
    TC track (movement "characteristics for every 5 degree lat. bin and epsilon distribution)

    For now, will randomly perturb parameters from observed data
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import truncnorm
import os
from PlotMapData import plotLatLonGridData

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
def getObservedGenesisLocations(basin, month):

    basins=['EP','NA','NI','SI','SP','WP']

    idx = basins.index(basin)

    grid_copy = np.loadtxt(os.path.join(__location__, "STORM", 'GRID_GENESIS_MATRIX_' + str(idx) + '_' + str(month) + '.txt'))

    return grid_copy

def randomizedGenesisLocationMatrices(basin, monthlist, scale=1):
    '''
    Randomly perturb the observed genesis locations.
    Note: the current randomization is not based on any actual characteristics the data should have,
    it will need to be updated later

    :param monthlist: a list of months to create genesis location matrices for
    :param basin:
    :param scale: the scale used to perturb the data
    :return:
    '''

    genesisLocationMatrices = {}

    for month in monthlist:
        grid = getObservedGenesisLocations(basin, month)

        a, b = ( - 1)/3, (1.5 - 1)/3

        noise1 = truncnorm.rvs(a, b, loc=1, scale=3, size=grid.shape)

        noise2 = np.random.uniform(np.nanmin(grid), np.nanmax(grid), size=grid.shape)
        blurred_noise1 = gaussian_filter(noise1, 2)
        blurred_noise2 = gaussian_filter(noise2, 2)
        genesisLocationMatrices[month] = grid*blurred_noise1 + blurred_noise2

        plotLatLonGridData(genesisLocationMatrices[month], 1, basin=basin)

    return genesisLocationMatrices


def getMovementCoefficientData():
    constants_all=np.load(os.path.join(__location__,'STORM', 'TRACK_COEFFICIENTS.npy'),allow_pickle=True,encoding='latin1').item()

    return constants_all

def randomizedMovementCoefficients():

    # for now, just perturb the calculated track coefficients. it will be faster than refitting the lsq regressions.
    coefficientData = getMovementCoefficientData()

    for key, val in coefficientData.items():

        coefficientData[key] += np.random.normal(0, 1, np.array(val).shape)

        # can use slices for this probably
        for i in range(0, 11):
            coefficientData[key][i][6] = np.abs(coefficientData[key][i][6])
            coefficientData[key][i][8] = np.abs(coefficientData[key][i][8])
            coefficientData[key][i][10] = np.abs(coefficientData[key][i][10])
            coefficientData[key][i][12] = np.abs(coefficientData[key][i][12])

    return coefficientData


def generateInputParameters(basin, monthslist):
    """
    Create a genesis probability map and tc track movement distribution

    :return:
    """

    return randomizedGenesisLocationMatrices(basin, monthslist), randomizedMovementCoefficients()

randomizedGenesisLocationMatrices('SP', [1])


