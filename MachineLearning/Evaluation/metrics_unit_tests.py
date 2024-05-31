from MachineLearning.Evaluation.metrics import *
import unittest
from MachineLearning.Evaluation.relative_change_metrics import *
from MachineLearning.Evaluation.model_statistics import compute_ensemble_statistics
from MachineLearning.Evaluation.evaluation_utils import get_grid_cell, get_site_values, sites, get_many_site_values
from MachineLearning.Evaluation.site_metrics import site_mean_squared_error, site_squared_error
from MachineLearning.Evaluation.evaluation_testing_utils import *

## run by calling  pytest metrics_unit_tests.py::TestSiteMetrics in this directory

class TestSiteMetrics(unittest.TestCase):

    def get_test_statistics_and_metrics(self):

        outputs, predictions = get_outputs_and_predictions()
        storm_statistics = compute_ensemble_statistics(outputs)
        unet_statistics = compute_ensemble_statistics(predictions)

        all_metrics = compute_metrics(outputs, predictions, storm_statistics, unet_statistics, "Custom-UNet")

        return outputs, predictions, storm_statistics, unet_statistics, all_metrics

    """ Test Metric Code """

    def test_get_grid_cell(self):
        # test that the get grid cell function works properly for two different resolutions

        self.assertEqual(get_grid_cell(-60,135, 0.5), (0, 0))
        self.assertEqual(get_grid_cell(-60, 136.1, 0.5), (0, 2))
        self.assertEqual(get_grid_cell(-60, 136.1, 1), (0, 1))

    def test_get_grid_cell_out_of_basin(self):

        with self.assertRaises(Exception):
            get_grid_cell(-61, 4, 0.5)


    def test_get_site_values(self):

        # test that the get_site_values function only returns non-zero values for non-zero cells
        test_grid = np.zeros((210,110))

        site_vals = get_site_values(test_grid)

        self.assertEqual(site_vals.tolist(), [0 for i in range(len(site_vals))])

        cell = get_grid_cell(-9.81386294, 160.1563795, 0.5)
        print(cell)
        test_grid[*cell] = 1
        site_vals = get_site_values(test_grid)

        self.assertEqual(site_vals[0], 1)

        for i in range(1, len(site_vals)):
            if site_vals[i] != 0:
                self.assertEqual(site_vals[i], 1)
                self.assertEqual(get_grid_cell(*sites.sites[i], 0.5), cell)

    def test_site_se(self):
        ground_outputs, model_outputs = get_outputs_and_predictions()

        site_se_1 = site_squared_error(model_outputs[0], ground_outputs[0])

        self.assertEqual(site_se_1[0], 169)
        self.assertEqual(site_se_1[1], 0)

    def test_site_mse(self):
        ground_outputs, model_outputs = get_outputs_and_predictions()

        site_mse_1 = site_mean_squared_error(model_outputs[0], ground_outputs[0])

        n_non_zero = np.count_nonzero(site_mse_1)

        n_in_gridcell = 0
        non_zero_cell = (100, 50)
        for site in sites.sites:
            if get_grid_cell(*site, 0.5) == non_zero_cell:
                n_in_gridcell += 1

        self.assertEqual(site_mse_1, 169 * n_in_gridcell / 542)

    def test_relative_change(self):
        outputs, predictions = get_outputs_and_predictions()
        change_map = relative_change(outputs[0], outputs[1])

        self.assertAlmostEquals(change_map[100, 50], .2, 3)
        self.assertEqual(change_map[100, 51], 0)


    """ Test Ensemble Statistics """

    def test_ensemble_statistics_no_sites(self):
        outputs, predictions, storm_statistics, unet_statistics, all_metrics = self.get_test_statistics_and_metrics()
        self.assertEqual(storm_statistics["Quantiles"].shape, (5, 110, 210))
        self.assertEqual(unet_statistics["Quantiles"].shape, (5, 110, 210))

        self.assertEqual(storm_statistics["Quantiles"][:, 55, 45].tolist(), [5, 6.5, 8.5, 12, 18])
        self.assertEqual(unet_statistics["Quantiles"][:, 55, 45].tolist(), [10, 10, 10.5, 12.5, 17])

if __name__ == "__main__":
    unittest.main()

