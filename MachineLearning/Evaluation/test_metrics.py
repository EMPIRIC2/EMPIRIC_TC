import unittest

from MachineLearning.Evaluation.evaluation_testing_utils import (ALL_METRICS,
                                                                 STATISTICS_1,
                                                                 STATISTICS_2,
                                                                 TEST_GRIDS_1,
                                                                 TEST_GRIDS_2)
from MachineLearning.Evaluation.evaluation_utils import get_grid_cell, sites
from MachineLearning.Evaluation.relative_change_metrics import relative_change
from MachineLearning.Evaluation.site_metrics import (site_mean_squared_error,
                                                     site_squared_error)

outputs, predictions, storm_statistics, unet_statistics, all_metrics = (
    TEST_GRIDS_1,
    TEST_GRIDS_2,
    STATISTICS_1,
    STATISTICS_2,
    ALL_METRICS,
)


class TestMetrics(unittest.TestCase):
    def test_site_se(self):
        ground_outputs, model_outputs = TEST_GRIDS_1, TEST_GRIDS_2

        site_se_1 = site_squared_error(model_outputs[0], ground_outputs[0])

        self.assertEqual(site_se_1[0], 9)
        self.assertEqual(site_se_1[1], 0)

    def test_site_mse(self):
        site_mse_1 = site_mean_squared_error(predictions[0], outputs[0])

        n_in_gridcell = 0
        non_zero_cell = (9, 50)
        for site in sites.sites:
            if get_grid_cell(*site, 0.5) == non_zero_cell:
                n_in_gridcell += 1

        self.assertEqual(site_mse_1, 9 * n_in_gridcell / 542)

    def test_relative_change(self):
        change_map = relative_change(outputs[0], predictions[0])

        self.assertAlmostEqual(change_map[9, 50], 0.2, 3)
        self.assertEqual(change_map[9, 51], 0)

    def test_ensemble_statistics(self):
        self.assertEqual(storm_statistics["Quantiles"].shape, (5, 110, 210))
        self.assertEqual(unet_statistics["Quantiles"].shape, (5, 110, 210))

        self.assertEqual(
            storm_statistics["Quantiles"][:, 55, 45].tolist(), [5, 6.5, 8.5, 12, 18]
        )
        self.assertEqual(
            unet_statistics["Quantiles"][:, 55, 45].tolist(), [10, 10, 10.5, 12.5, 17]
        )


if __name__ == "__main__":
    unittest.main()
