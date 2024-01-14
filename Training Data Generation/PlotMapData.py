import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def BOUNDARIES_BASINS(idx):
    '''
    Copied from STORM model

    :param idx: basin index
    :return:
    '''
    if idx == 'EP':  # Eastern Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 180, 285
    if idx == 'NA':  # North Atlantic
        lat0, lat1, lon0, lon1 = 5, 60, 255, 359
    if idx == 'NI':  # North Indian
        lat0, lat1, lon0, lon1 = 5, 60, 30, 100
    if idx == 'SI':  # South Indian
        lat0, lat1, lon0, lon1 = -60, -5, 10, 135
    if idx == 'SP':  # South Pacific
        lat0, lat1, lon0, lon1 = -60, -5, 135, 240
    if idx == 'WP':  # Western Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 100, 180

    return lat0, lat1, lon0, lon1

def plotLatLonGridData(data, resolution, basin='SP'):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    # convert data into format for contour plot
    lats = [(len(data)-i) * resolution + lat0 for i in range(len(data))]
    lons = [i * resolution + lon0 for i in range(len(data[0]))]

    plt.contourf(lons, lats, data, 60, transform=ccrs.PlateCarree())

    plt.show()

