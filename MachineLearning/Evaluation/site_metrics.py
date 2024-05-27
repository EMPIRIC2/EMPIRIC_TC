import numpy as np
from .evaluation_utils import get_site_values

def site_se(ground, prediction):

    ## Note: This formulation weights cells according to how many sites they have

    pred_sites = get_site_values(prediction)
    true_sites = get_site_values(ground)

    return (pred_sites - true_sites) ** 2

def site_mse(ground, predictions):
    squared_errors = site_se(ground, predictions)
    return np.mean(squared_errors)

def total_site_mse(ground_outputs, model_outputs):
    assert len(ground_outputs) == len(model_outputs)

    mses = []
    for i in range(len(ground_outputs)):
        mses.append(site_mse(ground_outputs[i], model_outputs[i]))

    return np.mean(mses)
