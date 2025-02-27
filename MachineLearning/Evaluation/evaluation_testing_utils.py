""" Testing Utilities """
import numpy as np

import MachineLearning.Evaluation.model_statistics as model_statistics
from MachineLearning.Evaluation.metrics import compute_metrics


def create_test_grid(data):
    test_grid = np.zeros((110, 210))
    for point in data:
        test_grid[point[0], point[1]] = point[2]

    return test_grid


test_grid_parameters_1 = [
    [(55, 45, 10), (9, 50, 15), (55, 34, 5)],
    [(55, 45, 5), (9, 50, 20), (55, 34, 2)],
    [(55, 45, 7), (9, 50, 12), (55, 34, 3)],
    [(55, 45, 18), (9, 50, 0), (55, 34, 0)],
]

test_grid_parameters_2 = [
    [(55, 45, 11), (9, 50, 12), (55, 34, 6)],
    [(55, 45, 10), (9, 50, 17), (55, 34, 2)],
    [(55, 45, 10), (9, 50, 13), (55, 34, 4)],
    [(55, 45, 17), (9, 50, 2), (55, 34, 1)],
]


TEST_GRIDS_1 = [create_test_grid(params) for params in test_grid_parameters_1]
TEST_GRIDS_2 = [create_test_grid(params) for params in test_grid_parameters_2]

STATISTICS_1 = model_statistics.compute_ensemble_statistics("STORM", TEST_GRIDS_1)
STATISTICS_2 = model_statistics.compute_ensemble_statistics("UNet", TEST_GRIDS_2)

ALL_METRICS = compute_metrics(
    TEST_GRIDS_1, TEST_GRIDS_2, STATISTICS_1, STATISTICS_2, "Custom-UNet"
)
