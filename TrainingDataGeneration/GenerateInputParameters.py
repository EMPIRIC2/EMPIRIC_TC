"""
    Generate randomized input parameters for the STORM model that can be
    used to create a training dataset for machine learning.

    We only randomize the genesis location frequency.
"""
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import unittest

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def getObservedGenesisLocations(basin, month, matrix_path=None):
    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]

    idx = basins.index(basin)

    if matrix_path is None:
        matrix_path = os.path.join(
            __location__,
            "STORM",
            "GRID_GENESIS_MATRIX_" + str(idx) + "_" + str(month) + ".npy",
        )

    grid_copy = np.load(matrix_path)
    grid = np.nan_to_num(grid_copy)

    return grid

def randomizedGenesisLocationMatrices(rng, future_data, monthlist, include_historical=False, basin='SP'):
    """
    Randomly perturb the observed genesis locations.
    Note: the current randomization is not based on any
    actual characteristics the data should have,
    it will need to be updated later

    :param monthlist: a list of months to create genesis location matrices for
    :param basin:
    :param scale: the scale used to perturb the data
    :return:
    """
    models = ["CMCC-CM2-VHR4", "EC-Earth3P-HR", "CNRM-CM6-1-HR", "HadGEM3-GC31-HM"]
    genesis_location_matrices = {}

    if include_historical:
        weights = rng.dirichlet((1,1,1,1,1))
    else:
        weights = rng.dirichlet((1, 1, 1, 1))

    for month in monthlist:
        genesis_data_for_month = [
            np.nan_to_num(future_data[i][month]) for i in range(len(models))
        ]

        if include_historical:
            genesis_data_for_month.append(getObservedGenesisLocations(basin, month))
        grids = np.array(genesis_data_for_month)
        randomized_grid = (grids.T @ weights).T

        genesis_location_matrices[month] = randomized_grid

    return genesis_location_matrices, weights

def getMovementCoefficientData():
    constants_all = np.load(
        os.path.join(__location__, "STORM", "TRACK_COEFFICIENTS.npy"),
        allow_pickle=True,
        encoding="latin1",
    ).item()

    return constants_all

def generateInputParameters(
        future_data,
        monthslist,
        basin='SP',
        include_historical_genesis=False,
        constant_historical_inputs=False
):
    """
    Create a genesis probability map and tc track movement distribution

    :return:
    """
    # create a new seed to avoid generating the same inputs in every worker

    rng = np.random.Generator(np.random.PCG64DXSM())

    # if only using historical inputs, replace genesis with the historical
    if constant_historical_inputs:
        genesis_matrices = {}
        for month in monthslist:
            genesis_matrices[month] = getObservedGenesisLocations(basin, month)
        genesis_weightings = None

    else:
        genesis_matrices, genesis_weightings = randomizedGenesisLocationMatrices(
            rng,
            future_data,
            monthslist,
            include_historical=include_historical_genesis,
            basin=basin
        )

    return genesis_matrices, genesis_weightings, getMovementCoefficientData()

class InputGenerationTest(unittest.TestCase):
    def get_data(self):
        basin = "SP"

        monthsall = [
            [6, 7, 8, 9, 10, 11],
            [6, 7, 8, 9, 10, 11],
            [4, 5, 6, 9, 10, 11],
            [1, 2, 3, 4, 11, 12],
            [1, 2, 3, 4, 11, 12],
            [5, 6, 7, 8, 9, 10, 11],
        ]

        monthlist = monthsall[4]

        models = ["CMCC-CM2-VHR4", "EC-Earth3P-HR", "CNRM-CM6-1-HR", "HadGEM3-GC31-HM"]
        future_delta_files = [
            os.path.join(
                __location__,
                "InputData",
                "GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model),
            )
            for model in models
        ]

        future_data = [
            np.load(file_path, allow_pickle=True).item()[basin]
            for file_path in future_delta_files
        ]
        return future_data, monthlist

    def test_generate_input_parameters_random(self):
        # This test makes sure that two randomly generated
        # genesis matrices are actually different

        future_data, monthlist = self.get_data()

        (
            genesis_matrices,
            genesis_weightings,
            movement_coefficients,
        ) = generateInputParameters(future_data, monthlist)

        (
            genesis_matrices_2,
            genesis_weightings_2,
            movement_coefficients_2,
        ) = generateInputParameters(future_data, monthlist)

        for key in genesis_matrices.keys():
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_allclose,
                genesis_matrices[key],
                genesis_matrices_2[key],
            )

        self.assertEqual(movement_coefficients, movement_coefficients_2)


if __name__ == "__main__":
    unittest.main()
