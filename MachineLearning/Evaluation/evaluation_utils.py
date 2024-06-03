import numpy as np
import math
from HealthFacilities.getHealthFacilityData import Sites
import unittest

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

class TestEvaluationUtils(unittest.TestCase):
    def test_get_grid_cell(self):
        # test that the get grid cell function works properly for two different resolutions

        self.assertEqual(get_grid_cell(-60,135, 0.5), (0, 0))
        self.assertEqual(get_grid_cell(-60, 136.1, 0.5), (0, 2))
        self.assertEqual(get_grid_cell(-60, 136.1, 1), (0, 1))

    def test_get_grid_cell_out_of_basin(self):

        with self.assertRaises(Exception):
            get_grid_cell(-61, 4, 0.5)


    def test_get_site_values_zero(self):
        """
        test that the get_site_values function only returns 0s for 0 cells
        """

        test_grid = np.zeros((210,110))

        site_vals = get_site_values(test_grid)

        self.assertEqual(site_vals.tolist(), [0 for i in range(len(site_vals))])

    def test_get_site_values_non_zero(self):
        """
        test that get_site_values correctly returns non-zero value for site in non-zero cell
        """

        test_grid = np.zeros((210, 110))
        # these are the lat,lons for the first site
        cell = get_grid_cell(-9.81386294, 160.1563795, 0.5)

        test_grid[*cell] = 1
        site_vals = get_site_values(test_grid)

        # we confirm that the value of the first site has changed
        self.assertEqual(site_vals[0], 1)

        for i in range(1, len(site_vals)):
            if site_vals[i] != 0:

                # check that site value is 1
                self.assertEqual(site_vals[i], 1)

                # confirm that site is in the grid cell that was modified
                # no other sites should have non-zero values
                self.assertEqual(get_grid_cell(*sites.sites[i], 0.5), cell)

if __name__ == "__main__":
    unittest.main()
