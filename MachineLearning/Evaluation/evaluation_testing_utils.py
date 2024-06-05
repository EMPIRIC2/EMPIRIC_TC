""" Testing Utilities """
import numpy as np

def create_test_grid(data):
    test_grid = np.zeros((110, 210))
    for point in data:
        test_grid[point[0], point[1]] = point[2]

    return test_grid

test_grid_parameters_1 = [[(55, 45, 10), (100, 50, 15), (55, 34, 5)],
                        [(55, 45, 5), (100, 50, 20), (55, 34, 2)],
                        [(55, 45, 7), (100, 50, 12), (55, 34, 3)],
                        [(55, 45, 18), (100, 50, 0), (55, 34, 0)]
                        ]

test_grid_parameters_2 = [[(55, 45, 11), (100, 50, 12), (55, 34, 6)],
                        [(55, 45, 10), (100, 50, 17), (55, 34, 2)],
                        [(55, 45, 10), (100, 50, 13), (55, 34, 4)],
                        [(55, 45, 17), (100, 50, 2), (55, 34, 1)]]

TEST_GRIDS_1 = [create_test_grid(params) for params in test_grid_parameters_1]
TEST_GRIDS_2 = [create_test_grid(params) for params in test_grid_parameters_2]