import numpy as np
import os
import unittest

def get_gcm_genesis_maps(genesis_data_folder):
    """
    Load the gcm genesis maps

    @param genesis_data_folder: path to the folder containing the gcm genesis maps
    @return: array of the genesis maps generated from the gcm data
    """
    basin = "SP"

    models = ["CMCC-CM2-VHR4", "EC-Earth3P-HR", "CNRM-CM6-1-HR", "HadGEM3-GC31-HM"]
    future_delta_files = [
        os.path.join(
            genesis_data_folder,
            "GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model),
        )
        for model in models
    ]

    future_data = [
        np.load(file_path, allow_pickle=True).item()[basin]
        for file_path in future_delta_files
    ]

    return future_data

def get_genesis_map_from_weights(weights: list, gcm_maps: list):
    """
    @param weights: a list of 4 floats that sum to 1 representing a linear weighting between the gcm maps.
    @param gcm_maps: gcm maps from get_gcm_genesis_maps
    @return:
    """
    if len(weights) != 4 or sum(weights) != 1:
        raise Exception()

    monthlist =  [1, 2, 3, 4, 11, 12]

    genesis_location_matrices = {}

    for month in monthlist:
        genesis_data_for_month = [
            np.nan_to_num(gcm_maps[i][month]) for i in range(4)
        ]

        grids = np.array(genesis_data_for_month)
        randomized_grid = (grids.T @ np.array(weights)).T

        genesis_location_matrices[month] = randomized_grid

    genesis_matrix = np.array(
        [np.round(genesis_location_matrices[month], 1) for month in monthlist]
    )

    return genesis_matrix

class GenesisMapTest(unittest.TestCase):

    def test_get_genesis_maps(self):

        path = "../TrainingDataGeneration/InputData"
        genesis_maps = get_gcm_genesis_maps(path)
        inputs = get_genesis_map_from_weights([0.25, 0.25, 0.25, 0.25], genesis_maps)

        self.assertEqual(inputs.shape, (6, 55, 105))

    def test_get_genesis_maps_fails_wrong_weights(self):
        path = "../TrainingDataGeneration/InputData"
        genesis_maps = get_gcm_genesis_maps(path)
        with self.assertRaises(Exception):
            get_genesis_map_from_weights([0.3, 0.25, 0.25, 0.25], genesis_maps)