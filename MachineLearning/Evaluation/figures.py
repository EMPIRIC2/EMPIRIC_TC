import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader, natural_earth
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from HealthFacilities.getHealthFacilityData import Sites

from MachineLearning.Evaluation.evaluation_utils import (
    get_lat_lon_data_for_mesh, get_many_site_values, get_site_name, get_grid_cell)
from MachineLearning.Evaluation.site_metrics import (site_abs_error)

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
                        "Mean Cat 3-5 Landfalls/10 Years": site_outputs[j][i],
                        "Model": model,
                    }
                )

    df = pd.DataFrame(data)
    sns.boxplot(
        data=df, x="Site Name", y="Mean Cat 3-5 Landfalls/10 Years", hue="Model"
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
            vmax=1,
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

'''
def plot_example_outputs(all_outputs, save_path):

    n_examples = 5
    
    n_models = len(all_outputs.keys())
    fig, axes = plt.subplots(n_models, n_examples, figsize=(20,15), layout="compressed")
    
    model_n = 0
    
    max_val = np.max([list(outputs) for outputs in all_outputs.values()])
    
    for model, outputs in all_outputs.items():

        for i in range(n_examples):
            ax = axes[model_n, i]
            
            ax.set_xticks([])
            ax.set_yticks([])

            f = ax.imshow(outputs[i], vmin=0, vmax=max_val)

        model_n += 1

    for model_idx, model in enumerate(all_outputs.keys()):
        ax = axes[model_idx, 0]
        ax.set_ylabel(model, rotation=0, ha='right')
        

    fig.colorbar(f, ax=axes)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else: 
        plt.show()
''' 

       

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

'''
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
        plt.show()'''


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
            site_errors = site_abs_error(outputs[i], ground_outputs[i])
            for j in range(site_errors.shape[0]):
                box_plot_data.append(
                    {
                        "Site Absolute Error": site_errors[j],
                        "Test Example": i,
                        "Model": model,
                    }
                )

    df = pd.DataFrame(box_plot_data)
    b = sns.boxplot(df, x="Test Example", y="Site Absolute Error", hue="Model")
    b.set_xticklabels(b.get_xticks(), size=5)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def plot_example_outputs(all_outputs, save_path=None, sites=None, n_examples=3):

    # Add transparent land by creating a custom feature
    land_shp = natural_earth(resolution='110m', category='physical', name='land')
    land_feature = ShapelyFeature(Reader(land_shp).geometries(),
                                  ccrs.PlateCarree(), facecolor='lightgray', edgecolor='face', alpha=0.5)
    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    
    # Add countries boundaries and labels
    countries_shp = natural_earth(resolution='50m', category='cultural', name='admin_0_countries')
    countries_feature = ShapelyFeature(Reader(countries_shp).geometries(), ccrs.PlateCarree(),
                                       facecolor='none', edgecolor='black', linewidth=1, alpha=1)
    
    n_models = len(all_outputs.keys())
    fig, axes = plt.subplots(n_models, n_examples, figsize=(20,15), layout="compressed",  subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    model_n = 0
    
    max_val = np.max([list(outputs) for outputs in all_outputs.values()])
    
    for model, outputs in all_outputs.items():

        for i in range(n_examples):
            ax = axes[model_n, i]

            ax.add_feature(land_feature)
            ax.add_feature(countries_feature)
            #ax.add_feature(states_provinces, edgecolor='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            
            val = outputs[i]
            
            # Plot gradient for model prediction on top of the land feature
            lonplot2 = np.linspace(135, 240, val.shape[1])
            latplot2 = np.linspace(-5, -60, val.shape[0])

            
            # Use a continuous colormap
            contour = ax.contourf(lonplot2, latplot2, val, levels=100, cmap='summer', transform=ccrs.PlateCarree())
            
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=1, color='white', alpha=0.3, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            if sites is not None:
                # s
                y, x = zip(*sites)
                
                x = [x[i] - 180 for i in range(len(x))]
                ax.scatter(x, y, s=10, c='m')
                ax.set_extent([150, 185, -23, -5], crs=ccrs.PlateCarree())
                
            else:
                ax.set_extent([135, 240, -60, -5], crs=ccrs.PlateCarree())
            
        model_n += 1

    for model_idx, model in enumerate(all_outputs.keys()):
        ax = axes[model_idx, 0]
        ax.set_ylabel(model, rotation=0, ha='right')
        
    fig.colorbar(contour, ax=axes)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else: 
        plt.show()

## Example hospital coordinates (lat, lon)

EXAMPLE_HOSPITALS = [{
    "name": "Vaiola",
    "coordinates": (-21.15515, 184.781183)
}, {
    "name": "Honiara National Referral Hospital",
    "coordinates": (-9.43531212, 159.96918117)
}, {
    "name": "Vila Central",
    "coordinates": (-17.7423312961028, 168.321078380767)
}]

def make_example_site_histogram_figures(all_outputs, save_path=None):

    n_examples = len(EXAMPLE_HOSPITALS)
    fig, axes = plt.subplots(1, n_examples, figsize=(20,15), layout="compressed")
    
    df = pd.DataFrame(columns = ['Mean Decadal Landfalls', 'Hospital', 'Model'])
    
    for i in range(len(EXAMPLE_HOSPITALS)):
        cell = get_grid_cell(*EXAMPLE_HOSPITALS[i]["coordinates"], 0.5)
        hospital_name = EXAMPLE_HOSPITALS[i]["name"]
        model_names = []
        for model, outputs in all_outputs.items():
            for output in outputs:
                df.loc[len(df.index)] = [output[cell], hospital_name, model]

    df.to_pickle(os.path.join(os.path.dirname(save_path), "histogram_data.pkl"))
    
    ## how to plot the data
    g = sns.displot(data=df,
                    col='Hospital',
                    hue='Model',
                    x='Mean Decadal Landfalls',
                    element='step', 
                    common_bins=False,
                    common_norm=False,
                    facet_kws={'sharex': False}
                   )
    fig = g.fig

    if save_path is not None:
        fig.savefig(save_path)        

def make_collective_model_figures(all_outputs, all_statistics, save_dir):
    """
    Make figures that involve all of the models

    @param all_outputs: dict containing the outputs of every model {model: outputs}
    @param all_statistics: list of statistic objects for every model (including storm}
    @param save_dir: folder to save data in
    """

    all_site_outputs = {
        model: get_many_site_values(outputs) for model, outputs in all_outputs.items()
    }

    example_site_ensemble_boxplot_figure(
        all_site_outputs, os.path.join(save_dir, "site_ensemble_boxplot.png")
    )

    plot_quantile_maps(all_statistics, os.path.join(save_dir, "quantile_maps.png"))

    ground_outputs = all_outputs["STORM"]
    models_outputs = {k: i for k, i in all_outputs.items() if k != "STORM"}
    plot_example_site_boxplot(
        ground_outputs,
        models_outputs,
        10,
        os.path.join(save_dir, "example_site_boxplots.png"),
    )

    sites = Sites()
    
    plot_example_outputs(all_outputs, os.path.join(save_dir, "example_outputs_sites.png"), sites=sites.sites, n_examples=3)

    plot_example_outputs(all_outputs, os.path.join(save_dir, "example_outputs.png"), n_examples=3)
    
    make_example_site_histogram_figures(all_outputs, os.path.join(save_dir, "site_histogram_outputs.png"))

def make_single_model_figures(
    metrics,
    save_dir,
):
    """
    Creates all figures for a single network

    @param metrics: metrics dict for the network for which to plot figures
    @param save_folder:
    @return:
    """

    ks_statistic_map(metrics, os.path.join(save_dir, "ks_statistics.png"))
    '''top_relative_error_maps(
        metrics["Relative Error Examples"],
        os.path.join(save_dir, "worst_relative_errors.png"),
    )'''
