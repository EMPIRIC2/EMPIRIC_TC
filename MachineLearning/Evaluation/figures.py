import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn_image as isns

from MachineLearning.Evaluation.evaluation_utils import (
    get_lat_lon_data_for_mesh, get_many_site_values, get_site_name)
from MachineLearning.Evaluation.site_metrics import site_squared_error


def example_site_ensemble_boxplot_figure(all_site_outputs, save_path=None):
    """
    Outputs boxplot showing the distributions
    of values at the first 10 sites across all the outputs
    Compares all the models over each site.

    all_site_outputs: dict of {model_name: site_outputs}
    """

    data = []
    for i in range(10):
        for model, site_outputs in all_site_outputs.items():
            for j in range(len(site_outputs)):
                data.append(
                    {
                        "Site Name": get_site_name(i),
                        "Mean Cat 0-2 Landfalls/10 Years": site_outputs[j][i],
                        "Model": model,
                    }
                )

    df = pd.DataFrame(data)
    sns.boxplot(
        data=df, x="Site Name", y="Mean Cat 0-2 Landfalls/10 Years", hue="Model"
    )
    plt.xticks(rotation=90)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def plot_quantile_maps(models_statistics, save_path=None):
    """
    Outputs images of the quantiles of both the ground
    outputs and the model outputs. Quantiles are 0, .25, .5, .75, 1.
    The figure has two rows of 5 images. The top row is
    the quantiles from ground, increasing from left to right.
    The bottom row is the quantiles from model,
    also increasing from left to right.

    If called with a save path, it saves the figure to
    that path. Otherwise, it displays the image.
    """

    images = np.empty((0, 110, 210))

    for model_statistic in models_statistics:
        images = np.concatenate([images, model_statistic["Quantiles"]])

    cols = ["Quantile: {}".format(col) for col in [0, 0.25, 0.5, 0.75, 1]]

    row_names = [model_statistics["Model"] for model_statistics in models_statistics]
    rows = ["{}".format(row) for row in row_names]
    fig, axes = plt.subplots(
        nrows=len(models_statistics),
        ncols=5,
        figsize=(13, 5),
        layout="compressed",
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )

    axes = axes.flatten()
    for i in range(len(images)):
        ax = axes[i]
        lats, lons = get_lat_lon_data_for_mesh(images[i], resolution=0.5)

        ax.coastlines("10m")

        ax.set_xticks([])
        ax.set_yticks([])
        cm = ax.pcolormesh(
            lons,
            lats,
            images[i],
            transform=ccrs.PlateCarree(central_longitude=180),
            cmap="summer",
            vmin=0,
            vmax=10,
        )

    for ax, col in zip(axes[:5], cols):
        ax.set_title(col, pad=15)

    for ax, row in zip(axes[::5], rows):
        ax.set_ylabel(row, rotation=0, labelpad=40)

    fig.colorbar(cm, ax=axes)

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def ks_statistic_map(metrics, save_path=None):
    """
    Outputs a map of the KS statistic
    """
    ks = metrics["Kolmogorov-Smirnov"]
    fig = plt.figure(
        "Kolmogorov-Smirnov Statistic of STORM vs {}".format(metrics["Model"])
    )
    fig.suptitle("Kolmogorov-Smirnov Statistic of STORM vs {}".format(metrics["Model"]))
    lats, lons = get_lat_lon_data_for_mesh(ks, resolution=0.5)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines()

    im = plt.pcolormesh(
        lons,
        lats,
        ks,
        transform=ccrs.PlateCarree(central_longitude=180),
        vmin=0,
        vmax=1,
        cmap="summer",
    )
    plt.colorbar(im, fraction=0.025, pad=0.04)

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def metrics_df(all_model_metrics):
    """
    Creates a dataframe for metrics. Used to save the metrics in a latex table.
    """
    for model_metrics in all_model_metrics:
        # remove this entry because it can't go into the DF
        model_metrics.pop("Kolmogorov-Smirnov")
        model_metrics.pop("Relative Error Examples")

    df = pd.DataFrame(all_model_metrics)

    return df


def top_relative_error_maps(top_error_maps, save_path=None):
    """
    Outputs maps showing the largest relative errors. Maximum of 10 images.
    """

    if len(top_error_maps) > 10:
        top_error_maps = top_error_maps[:10]

    g = isns.ImageGrid(top_error_maps, col_wrap=5, axis=0, vmin=0, vmax=2, cbar=True)
    g.fig.suptitle("Relative Error Maps with the largest Mean Squared Errors")
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def save_metrics_as_latex(all_model_metrics, save_path):
    """
    Save the metrics dataframe in a latex table.
    """
    df = metrics_df(all_model_metrics)
    with open(save_path, "w") as tf:
        tf.write(df.to_latex())


def plot_example_site_boxplot(
    ground_outputs, models_outputs, n_examples, save_path=None
):
    """
    Outputs a boxplot showing n_examples
    distributions of site errors for different outputs.
    """
    box_plot_data = []

    for model, outputs in models_outputs.items():
        for i in range(n_examples):
            site_errors = site_squared_error(outputs[i], ground_outputs[i])
            for j in range(site_errors.shape[0]):
                box_plot_data.append(
                    {
                        "Site Squared Error": site_errors[j],
                        "Test Example": i,
                        "Model": model,
                    }
                )

    df = pd.DataFrame(box_plot_data)
    b = sns.boxplot(df, x="Test Example", y="Site Squared Error", hue="Model")
    b.set_xticklabels(b.get_xticks(), size=5)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def make_collective_model_figures(all_outputs, all_statistics, save_folder):
    all_site_outputs = {
        model: get_many_site_values(outputs) for model, outputs in all_outputs.items()
    }

    example_site_ensemble_boxplot_figure(
        all_site_outputs, os.path.join(save_folder, "site_ensemble_boxplot.png")
    )

    plot_quantile_maps(all_statistics, os.path.join(save_folder, "quantile_maps.png"))

    ground_outputs = all_outputs["STORM"]
    models_outputs = {k: i for k, i in all_outputs.items() if k != "STORM"}
    plot_example_site_boxplot(
        ground_outputs,
        models_outputs,
        10,
        os.path.join(save_folder, "example_site_boxplots.png"),
    )


def make_single_model_figures(
    ground_outputs,
    model_outputs,
    ground_statistics,
    model_statistics,
    metrics,
    save_folder,
):
    """
    Creates all figures for a single network

    @param ground_outputs:
    @param model_outputs:
    @param ground_statistics:
    @param model_statistics:
    @param metrics:
    @param save_folder:
    @return:
    """

    ks_statistic_map(metrics, os.path.join(save_folder, "ks_statistics.png"))
    top_relative_error_maps(
        metrics["Relative Error Examples"],
        os.path.join(save_folder, "worst_relative_errors.png"),
    )
