""" Testing Utilities """
import numpy as np
import model_statistics
import metrics

def create_test_grid(data):
    test_grid = np.zeros((110, 210))
    for point in data:
        test_grid[point[0], point[1]] = point[2]

    return test_grid

def get_test_grid_1():
    return create_test_grid([(55, 45, 10), (100, 50, 15), (55, 34, 5)])


def get_test_grid_2():
    return create_test_grid([(55, 45, 5), (100, 50, 20), (55, 34, 2)])


def get_test_grid_3():
    return create_test_grid([(55, 45, 7), (100, 50, 12), (55, 34, 3)])


def get_test_grid_4():
    return create_test_grid([(55, 45, 18), (100, 50, 0), (55, 34, 0)])


def get_test_grid_5():
    return create_test_grid([(55, 45, 11), (100, 50, 12), (55, 34, 6)])


def get_test_grid_6():
    return create_test_grid([(55, 45, 10), (100, 50, 17), (55, 34, 2)])


def get_test_grid_7():
    return create_test_grid([(55, 45, 10), (100, 50, 13), (55, 34, 4)])


def get_test_grid_8():
    return create_test_grid([(55, 45, 17), (100, 50, 2), (55, 34, 1)])


def get_outputs_and_predictions():
    # outputs
    test_grid_1 = get_test_grid_1()
    test_grid_2 = get_test_grid_2()
    test_grid_3 = get_test_grid_3()
    test_grid_4 = get_test_grid_4()
    outputs = [test_grid_1, test_grid_3, test_grid_4, test_grid_2]

    # predictions
    test_grid_5 = get_test_grid_5()
    test_grid_6 = get_test_grid_6()
    test_grid_7 = get_test_grid_7()
    test_grid_8 = get_test_grid_8()

    predictions = [test_grid_8, test_grid_5, test_grid_6, test_grid_7]

    return outputs, predictions

def get_test_statistics_and_metrics():

    outputs, predictions = get_outputs_and_predictions()
    storm_statistics = model_statistics.compute_ensemble_statistics(outputs)
    unet_statistics = model_statistics.compute_ensemble_statistics(predictions)

    all_metrics = metrics.compute_metrics(outputs, predictions, storm_statistics, unet_statistics, "Custom-UNet")

    return outputs, predictions, storm_statistics, unet_statistics, all_metrics
