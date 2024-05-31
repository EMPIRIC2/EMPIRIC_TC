from MachineLearning.Evaluation.metrics import *
import unittest
from MachineLearning.Evaluation.figures import *
from MachineLearning.Evaluation.relative_change_metrics import *
from MachineLearning.Evaluation.model_statistics import compute_ensemble_statistics
from MachineLearning.Evaluation.evaluation_utils import get_grid_cell, get_site_values, sites, get_many_site_values
from MachineLearning.Evaluation.site_metrics import site_mean_squared_error

## run by calling  pytest metrics_unit_tests.py::TestSiteMetrics in this directory
class TestSiteMetrics(unittest.TestCase):

    """ Testing Utilities """
    def create_test_grid(self, data):
        test_grid = np.zeros((110, 210))
        for point in data:
            test_grid[point[0], point[1]] = point[2]

        return test_grid
    def get_test_grid_1(self):
        return self.create_test_grid([(55, 45, 10), (100, 50, 15), (55, 34, 5)])

    def get_test_grid_2(self):
        return self.create_test_grid([(55, 45, 5), (100, 50, 20), (55, 34, 2)])

    def get_test_grid_3(self):
        return self.create_test_grid([(55, 45, 7), (100, 50, 12), (55, 34, 3)])

    def get_test_grid_4(self):
        return self.create_test_grid([(55, 45, 18), (100, 50, 0), (55, 34, 0)])

    def get_test_grid_5(self):
        return self.create_test_grid([(55, 45, 11), (100, 50, 12), (55, 34, 6)])

    def get_test_grid_6(self):
        return self.create_test_grid([(55, 45, 10), (100, 50, 17), (55, 34, 2)])

    def get_test_grid_7(self):
        return self.create_test_grid([(55, 45, 10), (100, 50, 13), (55, 34, 4)])

    def get_test_grid_8(self):
        return self.create_test_grid([(55, 45, 17), (100, 50, 2), (55, 34, 1)])

    def get_outputs_and_predictions(self):
        # outputs
        test_grid_1 = self.get_test_grid_1()
        test_grid_2 = self.get_test_grid_2()
        test_grid_3 = self.get_test_grid_3()
        test_grid_4 = self.get_test_grid_4()
        outputs = [test_grid_1, test_grid_3, test_grid_4, test_grid_2]

        # predictions
        test_grid_5 = self.get_test_grid_5()
        test_grid_6 = self.get_test_grid_6()
        test_grid_7 = self.get_test_grid_7()
        test_grid_8 = self.get_test_grid_8()

        predictions = [test_grid_8, test_grid_5, test_grid_6, test_grid_7]

        return outputs, predictions

    """ Test Metric Code """

    def test_site_se(self):
        ground_outputs, model_outputs = self.get_outputs_and_predictions()

        site_se_1 = site_squared_error(model_outputs[0], ground_outputs[0])

        self.assertEqual(site_se_1[0], 169)
        self.assertEqual(site_se_1[1], 0)

    def test_site_mse(self):
        ground_outputs, model_outputs = self.get_outputs_and_predictions()

        site_mse_1 = site_mean_squared_error(model_outputs[0], ground_outputs[0])

        n_in_gridcell = 0
        non_zero_cell = (100, 50)
        for site in sites.sites:
            if get_grid_cell(*site, 0.5) == non_zero_cell:
                n_in_gridcell += 1

        self.assertEqual(site_mse_1, 169 * n_in_gridcell / 542)

    def test_relative_change(self):
        outputs, predictions = self.get_outputs_and_predictions()
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

