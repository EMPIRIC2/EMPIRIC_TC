import unittest
from MachineLearning.Evaluation.evaluation_testing_utils import TEST_GRIDS_1, TEST_GRIDS_2, STATISTICS_1, STATISTICS_2, ALL_METRICS
from MachineLearning.Evaluation.relative_change_metrics import compute_changes_between_2_samples
from MachineLearning.Evaluation.figures import example_site_ensemble_boxplot_figure, plot_example_site_boxplot, ks_statistic_map, plot_quantile_maps, top_relative_error_maps
from MachineLearning.Evaluation.evaluation_utils import get_many_site_values

outputs, predictions, storm_statistics, unet_statistics, all_metrics = TEST_GRIDS_1, TEST_GRIDS_2, STATISTICS_1, STATISTICS_2, ALL_METRICS

class TestFigures(unittest.TestCase):
    """ Figure Tests """
    """ These will open example figures that must be closed for tests to complete """
    def test_ensemble_boxplot(self):
        example_site_ensemble_boxplot_figure({"STORM": get_many_site_values(outputs), "UNet": get_many_site_values(predictions)})

    def test_example_site_error_boxplot(self):
        plot_example_site_boxplot(outputs, predictions, 4)

    def test_ks_statistics_map(self):
        ks_statistic_map(all_metrics)

    def test_quantile_maps(self):
        plot_quantile_maps(storm_statistics, unet_statistics)

    def test_relative_change_error_map(self):
        error_map, total_error = compute_changes_between_2_samples(outputs, predictions, 0, 1)
        top_relative_error_maps(top_error_maps=[error_map, error_map])

if __name__ == "__main__":
    unittest.main()