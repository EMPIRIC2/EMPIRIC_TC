import numpy as np
from sklearn.metrics import mean_squared_error

from MachineLearning.Evaluation.evaluation_utils import (get_many_site_values,
                                                         get_site_values)


def site_squared_error(output1, output2):
    """
    Calculates the squared error at each health facility site
    Note that it will include sites that are in the same grid cell

    :param output1: a grid (2d array) output by one of the models
    :param output2: a grid (2d array) output by one of the models
    """

    # get numpy arrays of the output values for each health facility
    output1_sites = get_site_values(output1)
    output2_sites = get_site_values(output2)

    return (output1_sites - output2_sites) ** 2


def site_abs_error(output1, output2):
    """
    Calculates the absolute error at each health facility site
    Note that it will include sites that are in the same grid cell

    :param output1: a grid (2d array) output by one of the models
    :param output2: a grid (2d array) output by one of the models
    """

    # get numpy arrays of the output values for each health facility
    output1_sites = get_site_values(output1)
    output2_sites = get_site_values(output2)

    return np.abs(output2_sites - output1_sites)


def site_mean_squared_error(output1, output2):
    """
    Calculates the mean squared error at the health facility locations
    """
    squared_errors = site_squared_error(output1, output2)
    return np.mean(squared_errors, axis=0)


def total_site_mean_squared_error(outputs1, outputs2):
    """
    Calculates the mean of the site mean squared errors for the entire outputs dataset.

    :param: outputs1: list of grids of cyclone counts output by model 1
    :param: outputs2: list of grids of cyclone counts output by model 2

    :returns: the mean squared error at the
    health facility locations averaged over all the output examples
    """

    assert len(outputs1) == len(outputs2)

    outputs1_sites = get_many_site_values(outputs1)
    outputs2_sites = get_many_site_values(outputs2)

    return mean_squared_error(outputs1_sites, outputs2_sites)
