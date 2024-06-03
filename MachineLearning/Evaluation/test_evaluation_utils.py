import unittest
from MachineLearning.Evaluation.evaluation_utils import get_grid_cell, get_site_values, sites
import numpy as np

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
