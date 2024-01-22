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
import os
import ray
import secrets

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
def getObservedGenesisLocations(basin, month, matrix_path=None):

    basins=['EP','NA','NI','SI','SP','WP']

    idx = basins.index(basin)

    if matrix_path is None:
        matrix_path = os.path.join(__location__, "STORM", 'GRID_GENESIS_MATRIX_' + str(idx) + '_' + str(month) + '.npy')

    grid_copy = np.load(matrix_path)

    return grid_copy

def randomizedGenesisLocationMatrices(rng, basin, monthlist, scale=1):
    '''
    Randomly perturb the observed genesis locations.
    Note: the current randomization is not based on any actual characteristics the data should have,
    it will need to be updated later

    :param monthlist: a list of months to create genesis location matrices for
    :param basin:
    :param scale: the scale used to perturb the data
    :return:
    '''
    models = ['CMCC-CM2-VHR4', 'EC-Earth3P-HR', 'CNRM-CM6-1-HR', 'HadGEM3-GC31-HM']
    future_delta_files = [os.path.join(__location__, 'InputData', "GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model)) for model in models]

    future_data = [np.load(file_path, allow_pickle=True).item()[basin] for file_path in future_delta_files]
    genesis_location_matrices = {}

    for month in monthlist:

        weighted_factors_norm = rng.uniform(.9, 1.5)
        weights = rng.uniform(0, 1, size=(4,))

        normalized_weights = weighted_factors_norm * weights / np.sum(weights)
        future_data_for_month = [np.nan_to_num(future_data[i][month]) for i in range(len(models))]

        grids = np.array(future_data_for_month)
        randomized_grid = (grids.T @ normalized_weights).T

        genesis_location_matrices[month] = randomized_grid

    return genesis_location_matrices


def getMovementCoefficientData():
    constants_all=np.load(os.path.join(__location__,'STORM', 'TRACK_COEFFICIENTS.npy'),allow_pickle=True,encoding='latin1').item()

    return constants_all

def randomizedMovementCoefficients(rng):

    # for now, just perturb the calculated track coefficients. it will be faster than refitting the lsq regressions.

    movementCoefficientsFuture = [np.load('./InputData/JM_LONLATBINS_IBTRACSDELTA_{}.npy'.format(model)
                                          , allow_pickle=True)
                                  .item()['SP']
                                  for model in ['CMCC-CM2-VHR4','EC-Earth3P-HR','CNRM-CM6-1-HR','HadGEM3-GC31-HM']]

    # calculate statistics for each variable, per lat bin per month
    stds = np.std(movementCoefficientsFuture, axis=0)
    means = np.mean(movementCoefficientsFuture, axis=0)

    random_coefficients = rng.normal(means, stds)

    return {4: random_coefficients}


def generateInputParameters(basin, monthslist):
    """
    Create a genesis probability map and tc track movement distribution

    :return:
    """
    # create a new seed to avoid generating the same inputs in every worker

    rng = np.random.Generator(np.random.PCG64DXSM())

    return randomizedGenesisLocationMatrices(rng, basin, monthslist), randomizedMovementCoefficients(rng)
