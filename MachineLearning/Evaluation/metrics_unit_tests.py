import unittest
from MachineLearning.Evaluation.relative_change_metrics import relative_change
from MachineLearning.Evaluation.evaluation_utils import get_grid_cell, sites
from MachineLearning.Evaluation.site_metrics import site_mean_squared_error, site_squared_error
from MachineLearning.Evaluation.evaluation_testing_utils import get_outputs_and_predictions, get_test_statistics_and_metrics

class TestSiteMetrics(unittest.TestCase):
    """ Test Metric Code """
    def test_site_se(self):
        ground_outputs, model_outputs = get_outputs_and_predictions()

        site_se_1 = site_squared_error(model_outputs[0], ground_outputs[0])

        self.assertEqual(site_se_1[0], 169)
        self.assertEqual(site_se_1[1], 0)

    def test_site_mse(self):
        ground_outputs, model_outputs = get_outputs_and_predictions()

        site_mse_1 = site_mean_squared_error(model_outputs[0], ground_outputs[0])

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
        outputs, predictions, storm_statistics, unet_statistics, all_metrics = get_test_statistics_and_metrics()
        self.assertEqual(storm_statistics["Quantiles"].shape, (5, 110, 210))
        self.assertEqual(unet_statistics["Quantiles"].shape, (5, 110, 210))

        self.assertEqual(storm_statistics["Quantiles"][:, 55, 45].tolist(), [5, 6.5, 8.5, 12, 18])
        self.assertEqual(unet_statistics["Quantiles"][:, 55, 45].tolist(), [10, 10, 10.5, 12.5, 17])

if __name__ == "__main__":
    unittest.main()

