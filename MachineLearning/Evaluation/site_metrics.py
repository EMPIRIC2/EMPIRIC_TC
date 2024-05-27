import numpy as np
from .evaluation_utils import get_site_values

def site_se(ground, prediction):
    """
    Calculates the squared error at each health facility site
    Note that it will include sites that are in the same grid cell
    """
    pred_sites = get_site_values(prediction)
    true_sites = get_site_values(ground)

    return (pred_sites - true_sites) ** 2

def site_mse(ground, predictions):
    """
    Calculates the mean squared error at the health facility locations
    """
    squared_errors = site_se(ground, predictions)
    return np.mean(squared_errors)

def total_site_mse(ground_outputs, model_outputs):
    """
    Calculates the mean of the site mean squared errors for the entire outputs dataset.
    """
    assert len(ground_outputs) == len(model_outputs)

    mses = []
    for i in range(len(ground_outputs)):
        mses.append(site_mse(ground_outputs[i], model_outputs[i]))

    return np.mean(mses)
