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
import unittest
import numpy as np
import os

from scipy.stats import truncnorm
import secrets

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
def getObservedGenesisLocations(basin, month, matrix_path=None):

    basins=['EP','NA','NI','SI','SP','WP']

    idx = basins.index(basin)

    if matrix_path is None:
        matrix_path = os.path.join(__location__, "STORM", 'GRID_GENESIS_MATRIX_' + str(idx) + '_' + str(month) + '.npy')

    grid_copy = np.load(matrix_path)

    return grid_copy

def randomizedGenesisLocationMatrices(rng, future_data, monthlist, scale=1):
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
    genesis_location_matrices = {}

    for month in monthlist:

        weights = rng.dirichlet((1,1,1,1))

        future_data_for_month = [np.nan_to_num(future_data[i][month]) for i in range(len(models))]

        grids = np.array(future_data_for_month)
        randomized_grid = (grids.T @ weights).T

        genesis_location_matrices[month] = randomized_grid

    return genesis_location_matrices, weights


def getMovementCoefficientData():
    constants_all=np.load(os.path.join(__location__,'STORM', 'TRACK_COEFFICIENTS.npy'),allow_pickle=True,encoding='latin1').item()

    return constants_all

def randomizedMovementCoefficients(rng, movementCoefficientsFuture):

    # for now, just perturb the calculated track coefficients. it will be faster than refitting the lsq regressions.

    # calculate statistics for each variable, per lat bin per month
    stds = np.std(movementCoefficientsFuture, axis=0)
    means = np.mean(movementCoefficientsFuture, axis=0)

    # need to make sure that none of these are 0
    random_coefficients = rng.normal(means, stds)
    for j in range(6, 13, 2):

        for i in range(0, 11):
            loc = means[i][j]
            scale = stds[i][j]


            if scale == 0:
                scale = 10 ** -12

            a, b = (0 - loc) / scale, (100 - loc) / scale
            rv = truncnorm(a, b, loc=loc, scale=scale)
            random_coefficients[i][j] = rv.rvs(random_state=rng)

    return {4: random_coefficients}


def generateInputParameters(future_data, monthslist):
    """
    Create a genesis probability map and tc track movement distribution

    :return:
    """
    # create a new seed to avoid generating the same inputs in every worker

    rng = np.random.Generator(np.random.PCG64DXSM())
    genesis_matrices, genesis_weightings = randomizedGenesisLocationMatrices(rng, future_data, monthslist)
    return genesis_matrices, genesis_weightings, getMovementCoefficientData()

class InputGenerationTest(unittest.TestCase):

    def get_data(self):
        basin = 'SP'

        monthsall = [[6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], [4, 5, 6, 9, 10, 11], [1, 2, 3, 4, 11, 12],
                     [1, 2, 3, 4, 11, 12], [5, 6, 7, 8, 9, 10, 11]]

        monthlist = monthsall[4]

        models = ['CMCC-CM2-VHR4', 'EC-Earth3P-HR', 'CNRM-CM6-1-HR', 'HadGEM3-GC31-HM']
        future_delta_files = [
            os.path.join(__location__, 'InputData', "GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model))
            for model in models]

        future_data = [np.load(file_path, allow_pickle=True).item()[basin] for file_path in future_delta_files]
        return future_data, monthlist
    def test_generate_input_parameters_random(self):

        future_data, monthlist = self.get_data()

        genesis_matrices, genesis_weightings, movement_coefficients = generateInputParameters(future_data, monthlist)

        genesis_matrices_2, genesis_weightings_2, movement_coefficients_2 = generateInputParameters(future_data, monthlist)

        for key in genesis_matrices.keys():
            np.testing.assert_raises(AssertionError, np.testing.assert_allclose, genesis_matrices[key], genesis_matrices_2[key])

        self.assertEqual(movement_coefficients, movement_coefficients_2)

if __name__ == "__main__":
    unittest.main()