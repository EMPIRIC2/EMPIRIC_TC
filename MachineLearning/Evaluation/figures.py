import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn_image as isns
import numpy as np
import os

from .site_metrics import site_se
from .evaluation_utils import get_site_values_from_grid, get_site_name

def example_site_ensemble_boxplot_figure(all_site_outputs, save_path=None):
    '''
    Outputs boxplot showing the distributions of values at the first 10 sites across all the outputs
    Compares all the models over each site.

    all_site_outputs: dict of {model_name: site_outputs}
    '''

    data = []
    for i in range(10):
        for model, site_outputs in all_site_outputs.items():
            for j in range(len(site_outputs)):
                data.append({"Site Name": get_site_name(i), "Count": site_outputs[j][i], "Model": model})

    df = pd.DataFrame(data)
    sns.boxplot(data=df, x="Site Name", y="Count", hue="Model")

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_quantile_maps(ground_statistics, model_statistics, save_path=None):
    """
    Outputs images of the quantiles of both the ground outputs and the model outputs. Quantiles are 0, .25, .5, .75, 1.
    The figure has two rows of 5 images. The top row is the quantiles from ground, increasing from left to right.
    The bottom row is the quantiles from model, also increasing from left to right.

    If called with a save path, it saves the figure to that path. Otherwise, it displays the image.
    """

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
    """
    Outputs a map of the KS statistic
    """
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
    """
    Creates a dataframe for metrics. Used to save the metrics in a latex table.
    """
    for model_metrics in all_model_metrics:
        # remove this entry because it can't go into the DF
        model_metrics.pop("Kolmogorov-Smirnov")

    df = pd.DataFrame(all_model_metrics)

    return df

def top_relative_error_maps(top_error_maps, save_path=None):
    """
    Outputs maps showing the largest relative errors. Maximum of 10 images.
    """

    if len(top_error_maps) > 10: top_error_maps = top_error_maps[:10]

    g = isns.ImageGrid(top_error_maps, col_wrap=5, axis=0, vmin=0, vmax=16, cbar=True)
    if save_path is not None:
        plt.savefig(save_path)
    else: plt.show()

def save_metrics_as_latex(all_model_metrics, save_path):
    """
    Save the metrics dataframe in a latex table.
    """
    df = metrics_df(all_model_metrics)
    with open(save_path, 'w') as tf:
        tf.write(df.to_latex())

def plot_example_site_boxplot(ground_outputs, model_outputs, n_examples, save_path=None):
    """
    Outputs a boxplot showing n_examples distributions of site errors for different outputs.
    """
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
    ## master function to run all the figures for model Evaluation and visualization

    site_outputs = get_site_values_from_grid(ground_outputs)
    site_predictions = get_site_values_from_grid(model_outputs)

    example_site_ensemble_boxplot_figure({"STORM": site_outputs, "UNet": site_predictions}, os.path.join(save_folder, "site_ensemble_boxplot.png"))
    ks_statistic_map(metrics, os.path.join(save_folder, "ks_statistics.png"))
    plot_quantile_maps(ground_statistics, model_statistics, os.path.join(save_folder, "quantile_maps.png"))
    top_relative_error_maps(metrics["Relative Error Examples"], os.path.join(save_folder, "worst_relative_errors.png"))
    plot_example_site_boxplot(ground_outputs, model_outputs, 10, os.path.join(save_folder, "example_site_boxplots.png"))
