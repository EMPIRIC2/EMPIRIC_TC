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
    print(prediction.shape)
    rate = prediction[0][dist_index]
    print(rate)
    #std = np.diagonal(prediction[0][..., 1:])[dist_index]

    plt.figure()
    ax = plt.axes()
    print(real_outputs.shape)
    print(real_outputs[0, :, 0, dist_index])
    sample = pd.DataFrame(real_outputs[0, :, 0, dist_index], columns=["count"])
    sns.histplot(data=sample, x="count", stat='probability', color=sns_c[0], kde=False, ax=ax)

    # calculate the pdf
    x = np.arange(scipy.stats.poisson.ppf(0.0001, rate),
                  scipy.stats.poisson.ppf(0.9999, rate))

    y_pmf = scipy.stats.poisson.pmf(x, rate)

    ax.plot(x, y_pmf, 'bo', ms=8, label='poisson pmf')

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
    parser.add_argument('output_file', type=str,
                        help='path to file with the storm outputs in it')
    args = parser.parse_args()

    plot_site_distributions(args.file, args.output_file, 150)
