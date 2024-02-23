import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from HealthFacilities.getHealthFacilityData import Sites
import argparse

def plot_site_predictions(predictions_path):

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

    plot_site_predictions(args.file)