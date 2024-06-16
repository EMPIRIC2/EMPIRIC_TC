import math

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def BOUNDARIES_BASINS(idx):
    """
    Copied from STORM model

    :param idx: basin index
    :return:
    """
    if idx == "EP":  # Eastern Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 180, 285
    if idx == "NA":  # North Atlantic
        lat0, lat1, lon0, lon1 = 5, 60, 255, 359
    if idx == "NI":  # North Indian
        lat0, lat1, lon0, lon1 = 5, 60, 30, 100
    if idx == "SI":  # South Indian
        lat0, lat1, lon0, lon1 = -60, -5, 10, 135
    if idx == "SP":  # South Pacific
        lat0, lat1, lon0, lon1 = -60, -5, 135, 240
    if idx == "WP":  # Western Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 100, 180

    return lat0, lat1, lon0, lon1


def plotLatLonGridDataMultiple(
    datas, resolution, basin="SP", show=True, plot_names=None
):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=math.ceil(len(datas) / 2),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        figsize=(11, 8.5),
    )

    axs = axs.flatten()

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    for i, data in enumerate(datas):
        lats = [(len(data) - j) * resolution + lat0 for j in range(len(data))]
        lons = [j * resolution + lon0 - 180 for j in range(len(data[0]))]

        cs = axs[i].pcolormesh(
            lons, lats, data, transform=ccrs.PlateCarree(central_longitude=180)
        )

        if plot_names is not None and len(plot_names) == len(datas):
            axs[i].set_title(plot_names[i])

        axs[i].coastlines()

    for i in range(len(datas), math.ceil(len(datas) / 2) * 2):
        fig.delaxes(axs[i])

    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(
        bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.02, hspace=0.02
    )

    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
    # Draw the colorbar
    fig.colorbar(cs, cax=cbar_ax, orientation="horizontal")

    if show:
        plt.show()


def plotLatLonGridData(
    data,
    resolution,
    basin="SP",
    show=True,
    figure_name="",
    points=None,
    point_weight=None,
    boxes=None,
):
    plt.figure(figure_name)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines(resolution="10m")

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    # convert data into format for contour plot

    lats = [(len(data) - i) * resolution + lat0 for i in range(len(data))]

    lons = [j * resolution + lon0 - 180 for j in range(len(data[0]))]

    plt.pcolormesh(
        lons, lats, data, transform=ccrs.PlateCarree(central_longitude=180), vmin=-2
    )
    plt.colorbar()
            
    if boxes is not None:
        for box in boxes:
            rect = Rectangle((box[2] - 180, box[0]), box[3] - box[2], box[1] - box[0], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    if points is not None:
        assert len(points) == len(point_weight)

        y, x = zip(*points)
        x = [x[i] - 180 for i in range(len(x))]
        plt.scatter(x, y, s=point_weight, c=point_weight)

    if show is True:
        plt.show()

        
def showPlots():
    plt.show()
