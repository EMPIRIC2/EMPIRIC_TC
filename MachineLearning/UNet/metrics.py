import itertools

import numpy as np

from MachineLearning.dataset import get_dataset
from MachineLearning.UNet.unet import UNet
from HealthFacilities.getHealthFacilityData import Sites

import math
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn_image as isns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

########################################
#### Helper Functions and utilities ####
########################################

def getGridCell(lat, lon, resolution):
    '''
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    '''

    lat0, lat1, lon0, lon1 = -60, -5, 135, 240

    if lat < lat0 or lat >= lat1:
        raise Exception("lat must be within the basin")

    if lon < lon0 or lon >= lon1:
        raise Exception("lon must be within the basin")

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
        cell = getGridCell(*site, .5)
        squared_error = (pred[cell] - true[cell]) ** 2

        squared_errors.append(squared_error)
    return squared_errors

def site_mse(pred, true):
    squared_errors = site_se(pred, true)
    return np.mean(squared_errors)

def total_site_mse(ground_outputs, model_outputs):
    assert len(ground_outputs) == len(model_outputs)

    mses = []
    for i in range(len(ground_outputs)):
        mses.append(site_mse(ground_outputs[i], model_outputs[i]))

    return np.mean(mses)

def relative_change(a, b):
    """
    Returns the relative change from a to b.
    """

    return (a - b) / (a + 1e-3)

def _get_outputs(dataset):
    return dataset.map(lambda x, y: y)

def _get_inputs(dataset):
    return dataset.map(lambda x,y: x)

###############################################################
####### Relative Change Metrics between pairs of Inputs #######
###############################################################

def compute_changes_between_2_samples(ground_outputs, model_outputs, i, j):
    '''

    given two inputs and two models, compute the error in relative (percentage) change between two inputs, between the models
    i.e. this provides a measure of how well the model captures changes between two different inputs compared to the ground model (STORM)

    :param ground_outputs: the outputs from the ground truth model (STORM -> RISK)
    :param model_outputs: the outputs from the model being evaluated
    :param i: index of first example
    :param j: index of second example
    '''

    ground_output_1 = ground_outputs[i]
    ground_output_2 = ground_outputs[j]

    ground_change = relative_change(ground_output_1, ground_output_2)

    model_output_1 = model_outputs[i]
    model_output_2 = model_outputs[j]

    model_change = relative_change(model_output_1, model_output_2)

    error_map = ground_change - model_change

    return error_map, mean_squared_error(ground_change, model_change)

def compute_all_relative_change_pairs(ground_outputs, model_outputs):
    """
    Compute the error in relative output changes for all pairs outputs

    return: maps of the relative errors with the largest 10 mean squared errors and the total mean squared error (over all pair results)
    """

    pairs = itertools.combinations(range(len(ground_outputs)), 2)

    print("Number of Pairs: {}".format(len(list(pairs))))
    error_maps = []
    mean_squared_errors = []
    for pair in pairs:
        error_map, mse = compute_changes_between_2_samples(ground_outputs, model_outputs, *pair)
        error_maps.append(error_map)
        mean_squared_errors.append(mse)

    total_mse = np.mean(mean_squared_errors)

    n_examples = min(10, len(ground_outputs))
    largest_error_indices = np.argpartition(mean_squared_errors, -n_examples)[-n_examples:]
    top_error_maps = np.array(error_maps)[largest_error_indices]

    return top_error_maps, total_mse


############################################################
##### Statistics for Collections (ensembles) of inputs #####
############################################################

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
    ground_truth_sample_length = sample_shape[0] * sample_shape[1]
    ground_truths = np.reshape(ground_truths, (len(ground_truths), ground_truth_sample_length))
    predictions = np.reshape(predictions, (len(ground_truths), ground_truth_sample_length))

    # compute the kolmogorov smirnov statistic
    for i in range(len(ground_truths[0])):
        ks_statistics.append(scipy.stats.kstest(ground_truths[:,i], predictions[:,i]).statistic)

    # regrid the ks statistics
    ks_statistics = np.reshape(ks_statistics, sample_shape)
    return ks_statistics

###############
### Metrics ###
###############
def compute_metrics(ground_outputs, model_outputs, ground_statistics, model_statistics, model_name):

    top_relative_change_error_maps, mse = compute_all_relative_change_pairs(ground_outputs, model_outputs)

    # Compute metrics from the statistics
    metrics = {
        "Model": model_name,
        "Mean Absolute Quantile Error":  mean_absolute_error(ground_statistics["Quantiles"].flatten(), model_statistics["Quantiles"].flatten()),
        "Mean Squared Quantile Error":  mean_squared_error(ground_statistics["Quantiles"].flatten(), model_statistics["Quantiles"].flatten()),
        "Kolmogorov-Smirnov": KS_statistics(ground_outputs, model_outputs),
        "Relative Change Mean Squared Error": mse,
        "Relative Error Examples": top_relative_change_error_maps,
        "Site Mean Squared Error": total_site_mse(ground_outputs, model_outputs)
    }

    return metrics

###############
### Figures ###
###############

def example_site_ensemble_boxplot_figure(all_site_outputs, save_path=None):
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

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_quantile_maps(ground_statistics, model_statistics, save_path=None):
    ground_quantiles = ground_statistics["Quantiles"]

    model_quantiles = model_statistics["Quantiles"]
    images = np.array([ground_quantiles, model_quantiles])
    images = images.reshape((10, 110, 210))

    g = isns.ImageGrid(images, col_wrap=5, axis=0, vmin=0, vmax=16, cbar=True)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def ks_statistic_map(metrics, save_path = None):

    ks = metrics["Kolmogorov-Smirnov"]
    plt.title("Kolmogorov-Smirnov Statistic of ensemble outputs of STORM vs {}".format(metrics["Model"]))
    plt.imshow(ks)
    plt.colorbar()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    # TODO: improve figure

def metrics_df(all_model_metrics):

    for model_metrics in all_model_metrics:
        # this entry can't go into the DF
        model_metrics.pop("Kolmogorov-Smirnov")

    df = pd.DataFrame(all_model_metrics)

    return df

def top_relative_error_maps(top_error_maps, save_path=None):
    g = isns.ImageGrid(top_error_maps, col_wrap=5, axis=0, vmin=0, vmax=16, cbar=True)
    if save_path is not None:
        plt.savefig(save_path)
    else: plt.show()

def save_metrics_as_latex(all_model_metrics, save_path):

    df = metrics_df(all_model_metrics)
    with open(save_path, 'w') as tf:
        tf.write(df.to_latex())

def plot_example_site_boxplot(ground_outputs, model_outputs, n_examples, save_path=None):

    box_plot_data = []
    for i in range(n_examples):
        site_errors = site_se(model_outputs[i], ground_outputs[i])
        for site_error in site_errors:
            box_plot_data.append({"Site Squared Error": site_error, "Test Example": i})

    df = pd.DataFrame(box_plot_data)
    sns.boxplot(df, x="Test Example", y="Site Squared Error")

    if save_path is not None:
        plt.savefig(save_path)
    else: plt.show()

def make_figures(ground_outputs, model_outputs, ground_statistics, model_statistics, metrics, save_folder):
    ## master function to run all the figures for model evaluation and visualization

    site_outputs = get_site_values_from_grid(ground_outputs)
    site_predictions = get_site_values_from_grid(model_outputs)

    example_site_ensemble_boxplot_figure({"STORM": site_outputs, "UNet": site_predictions}, os.path.join(save_folder, "site_ensemble_boxplot.png"))
    ks_statistic_map(metrics, os.path.join(save_folder, "ks_statistics.png"))
    plot_quantile_maps(ground_statistics, model_statistics, os.path.join(save_folder, "quantile_maps.png"))
    top_relative_error_maps(metrics["Relative Error Examples"], os.path.join(save_folder, "worst_relative_errors.png"))
    plot_example_site_boxplot(ground_outputs, model_outputs, 10, os.path.join(save_folder, "example_site_boxplots.png"))

#####################################################################
#### Master Function to Compute All Metrics and Save All Figures ####
#####################################################################
def compute_test_statistics(data_folder, model_path, output_save_folder):

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

    storm_statistics = compute_ensemble_statistics(outputs)
    unet_statistics = compute_ensemble_statistics(predictions)

    metrics = compute_metrics(outputs, predictions, storm_statistics, unet_statistics, "UNet Custom")

    save_metrics_as_latex(metrics, os.path.join(output_save_folder, "metrics.tex"))

    make_figures(outputs, predictions, storm_statistics, unet_statistics, metrics, output_save_folder)
