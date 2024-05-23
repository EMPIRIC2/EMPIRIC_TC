import numpy as np

from MachineLearning.dataset import get_dataset
from MachineLearning.UNet.unet import UNet
from HealthFacilities.getHealthFacilityData import Sites

import math
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

### Define metrics to evaluate the model
def getGridCell(lat, lon, resolution):
    '''
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    '''

    lat0, lat1, lon0, lon1 = -60, -5, 135, 240

    if lat < lat0 or lat >= lat1:
        raise "lat must be within the basin"

    if lon < lon0 or lon >= lon1:
        raise "lon must be within the basin"

    latCell = math.floor((lat - lat0) * 1 / resolution)
    lonCell = math.floor((lon - lon0) * 1 / resolution)

    return latCell, lonCell


sites = Sites(1).sites

def get_site_values(grid):
    site_values = np.zeros((len(sites),))

    for i, site in enumerate(sites):
        cell = getGridCell(*site, .5)
        site_values[i] = grid[cell]

    return site_values

def site_se(pred, true):

    ## Note: This formulation weights cells according to how many sites they have

    squared_errors = []

    for site in sites:
        cell = getGridCell(*site, .5, 'SP')
        squared_error = (pred[cell] - true[cell]) ** 2

        squared_errors.append(squared_error)
    return squared_errors

def site_mse(pred, true):
    squared_errors = site_se(pred, true)
    return np.mean(squared_errors)

def _get_outputs(dataset):
    return dataset.map(lambda x, y: y)

def _get_inputs(dataset):
    return dataset.map(lambda x,y: x)

def get_variance(data):
    return np.var(data, axis=0)

def get_quantiles(data):
    return np.quantile(data, [0, 0.25,.5,.75,1], axis=0)

def get_site_values_from_grid(grids):
    site_outputs = []
    for output in grids:
        site_outputs.append(get_site_values(output))

    return site_outputs

def compute_ensemble_statistics(outputs):

    statistics = {
        "Quantiles": get_quantiles(outputs),
        "Mean": np.mean(outputs, axis=0)
    }

    return statistics

def absolute_quantile_errors(quantiles_a, quantiles_b):
    # quantiles have shape (n_quantiles, data_shape...)
    # don't reduce the data_shape here (e.g. if data was a grid, we want to get the abs quantile error for each grid cell)

    return np.mean(np.abs(quantiles_a - quantiles_b), axis=0)

def KS_statistics(ground_truths, predictions):
    # compute the kolmogorov smirnov statistic across the ensemble of data
    ks_statistics = []

    # flatten all but the first axis
    sample_shape = ground_truths[0].shape
    print(sample_shape)
    ground_truth_sample_length = sample_shape[0] * sample_shape[1]
    ground_truths = np.reshape(ground_truths, (len(ground_truths), ground_truth_sample_length))
    predictions = np.reshape(predictions, (len(ground_truths), ground_truth_sample_length))

    # compute the kolmogorov smirnov statistic
    for i in range(len(ground_truths[0])):
        ks_statistics.append(scipy.stats.kstest(ground_truths[:,i], predictions[:,i]).statistic)

    # regrid the ks statistics
    ks_statistics = np.reshape(ks_statistics, sample_shape)
    return ks_statistics

def compute_metrics(ground_truths, predictions, true_statistics, predicted_statistics, model_name):

    # Compute metrics from the statistics
    metrics = {
        "Model": model_name,
        "Mean Absolute Quantile Error":  mean_absolute_error(true_statistics["Quantiles"].flatten(), predicted_statistics["Quantiles"].flatten()),
        "Mean Squared Quantile Error":  mean_squared_error(true_statistics["Quantiles"].flatten(), predicted_statistics["Quantiles"].flatten()),
        "Kolmogorov-Smirnov": KS_statistics(ground_truths, predictions)
    }

    return metrics

def example_site_ensemble_boxplot_figure(all_site_outputs):
    '''
    all_site_outputs: dict of {model_name: site_outputs}
    '''

    data = []
    for i in range(10):
        for model, site_outputs in all_site_outputs.items():
            for j in range(len(site_outputs)):
                data.append({"Site Name": str(i), "Count": site_outputs[j][i], "Model": model})

    df = pd.DataFrame(data)
    sns.boxplot(data=df, x="Site Name", y="Count", hue="Model")
    plt.show()

def metrics_df(all_model_metrics):

    for model_metrics in all_model_metrics:
        # this entry can't go into the DF
        model_metrics.pop("Kolmogorov-Smirnov")

    df = pd.DataFrame(all_model_metrics)

    return df

def make_figures(outputs, predictions):
    ## master function to run all the figures for model evaluation and visualization

    site_outputs = get_site_values_from_grid(outputs)
    site_predictions = get_site_values_from_grid(predictions)

    example_site_ensemble_boxplot_figure({"STORM": site_outputs, "UNet": site_predictions})

def compute_test_statistics(data_folder, model_path, sample_size):

    genesis_size_default=(55, 105, 1)
    movement_size_default = (11, 13)

    test_data = get_dataset(data_folder, data_version=3, dataset='test', batch_size=32)

    model = UNet(genesis_size_default, movement_size_default, 1)

    model.load_weights(model_path)

    outputs = _get_outputs(test_data)
    inputs = _get_inputs(test_data)

    predictions = model.predict(
        inputs,
        batch_size=32,
        verbose=2,
        steps=1
    )

    #storm_statistics = compute_ensemble_statistics(outputs)
    #unet_statistics = compute_ensemble_statistics(predictions)

    make_figures(outputs, predictions)
