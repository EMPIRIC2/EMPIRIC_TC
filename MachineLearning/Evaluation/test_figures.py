import unittest

from MachineLearning.Evaluation.evaluation_testing_utils import (ALL_METRICS,
                                                                 STATISTICS_1,
                                                                 STATISTICS_2,
                                                                 TEST_GRIDS_1,
                                                                 TEST_GRIDS_2)
from MachineLearning.Evaluation.evaluation_utils import get_many_site_values
from MachineLearning.Evaluation.figures import (
    example_site_ensemble_boxplot_figure,
    ks_statistic_map,
    plot_example_site_boxplot,
    plot_quantile_maps,
    make_example_site_histogram_figures
)

from MachineLearning.Evaluation.relative_change_metrics import \
    compute_all_relative_change_pairs

outputs, predictions, storm_statistics, unet_statistics, all_metrics = (
    TEST_GRIDS_1,
    TEST_GRIDS_2,
    STATISTICS_1,
    STATISTICS_2,
    ALL_METRICS,
)


class TestFigures(unittest.TestCase):
    """Figure Tests"""

    """ These will open example figures that must be closed for tests to complete """

    def test_ks_statistics_map(self):
        ks_statistic_map(all_metrics)

    def test_quantile_maps(self):
        plot_quantile_maps(
            [
                storm_statistics,
                unet_statistics,
                unet_statistics,
                unet_statistics,
                unet_statistics,
                unet_statistics,
            ]
        )

if __name__ == "__main__":
    unittest.main()
