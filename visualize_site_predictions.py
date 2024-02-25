import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from HealthFacilities.getHealthFacilityData import Sites
import argparse
import seaborn as sns
import tensorflow_probability as tfp
import pandas as pd
import scipy
tfd = tfp.distributions

def plot_site_distributions(predictions_path, real_output_path, dist_index=0):

    prediction = np.load(predictions_path, allow_pickle=True)
    real_outputs = np.load(real_output_path, allow_pickle=True)

    sns.set_style(
        style='darkgrid',
        rc={'axes.facecolor': '.9', 'grid.color': '.8'}
    )
    sns.set_palette(palette='deep')
    sns_c = sns.color_palette(palette='deep')

    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 100

    mean = np.squeeze(prediction[0][..., :1])[dist_index]

    std = np.diagonal(prediction[0][..., 1:])[dist_index]

    plt.figure()
    ax = plt.axes()
    sample = pd.DataFrame(real_outputs[dist_index], columns=["count"])

    sns.histplot(data=sample, x="count", stat='density', color=sns_c[0], kde=False, ax=ax)

    # calculate the pdf
    x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
    x_pdf = np.linspace(x0, x1, 100)

    y_pdf = scipy.stats.norm.pdf(x_pdf, loc=mean, scale=std)

    ax.plot(x_pdf, y_pdf, 'r', lw=2, label='pdf')

    plt.show()

def plot_site_predictions_on_map(predictions_path):

    sites = Sites().sites

    prediction = np.load(predictions_path, allow_pickle=True)

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines(resolution='10m')
    print("plot")
    y, x = zip(*sites)

    x = [x[i] - 180 for i in range(len(x))]

    mean = np.squeeze(prediction[0][..., :1])

    std = np.diagonal(prediction[0][..., 1:])

    print(mean.shape)
    print(std.shape)

    smallest = mean - 2 * std
    largest = mean + 2 * std
    print(smallest[0], largest[0])
    plt.scatter(x, y, s=largest**2 * 16, c=largest**2 * 16)
    plt.scatter(x, y, s=mean**2 * 16, c=mean**2* 16)
    plt.scatter(x,y, s=smallest**2* 16, c=smallest**2* 16)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize site predictions on the map')
    parser.add_argument('file',  type=str,
                    help='path to file with the predictions in it')
    args = parser.parse_args()

    plot_site_distributions(args.file, 0)
