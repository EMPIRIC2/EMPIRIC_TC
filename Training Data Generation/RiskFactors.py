"""
    Take a list of storm tracks from storm and compute landfalls per month
"""
import numpy as np
import math
from geopy import distance
from scipy.interpolate import BSpline, make_interp_spline

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

def createMonthlyLandfallGrid(basin, resolution):
    '''
    Generate a grid to store TC monthly landfall statistics
    :param basin
    :param resolution: how many degrees each bin should be
    :return:
    '''
    # create lat lon grid (only needs to cover the basin) at 1 degree resolution
    # with [0] * 12 matrix in each

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    return np.zeros((
        math.ceil((lat1 - lat0) * 1/resolution),
        math.ceil((lon1 - lon0) * 1/resolution),
        12)
    )

def getGridCell(lat, lon, resolution, basin):
    '''
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    '''

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    if lat < lat0 or lat >= lat1:
        raise "lat must be within the basin"

    if lon < lon0 or lon >= lon1:
        raise "lon must be within the basin"

    latCell = math.floor((lat - lat0) * 1/resolution)
    lonCell = math.floor((lon - lon0) * 1/resolution)

    return latCell, lonCell

def getLatBoundsForCell(latCell, resolution, basin):
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    minimum = latCell * resolution + lat0
    return minimum, minimum+resolution
def getLonBoundsForCell(lonCell, resolution, basin):
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    minimum = lonCell * resolution + lon0
    return minimum, minimum+resolution

def getClosestLatInGridCellToStormCenter(latCell, storm_lat, resolution, basin):

    lat0, lat1 = getLatBoundsForCell(latCell, resolution, basin)

    # if the storm is at smaller latitude than the smallest in the cell,
    # the closest we can get is at the smallest latitude in the cell, etc.
    # if the storm is in the lat bounds of the grid cell, we can make the lat distance 0
    if storm_lat < lat0: return lat0
    if storm_lat > lat1: return lat1
    return storm_lat

def getClosestLonInGridCellToStormCenter(lonCell, storm_lon, resolution, basin):
    lon0, lon1 = getLonBoundsForCell(lonCell, resolution, basin)

    # same logic as getSmallestLatDistanceInGridCellToStormCenter
    if storm_lon < lon0: return lon0
    if storm_lon > lon1: return lon1
    return storm_lon

def checkCellTouchedByStorm(storm_lat, storm_lon, latCell, lonCell, rmax, resolution, basin):

    closest_lat = getClosestLatInGridCellToStormCenter(latCell, storm_lat, resolution, basin)
    closest_lon = getClosestLonInGridCellToStormCenter(lonCell, storm_lon, resolution, basin)

    if distance.distance((closest_lat, closest_lon), (storm_lat, storm_lon)).km <= rmax:
        return True

    return False

def getGridCellsTouchedByStormRMax(lat, lon, rmax, resolution, basin):


    # make bounding box based on rmax for which grid cells to iterate through
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    latitudeRmax = rmax/111 # approximate conversion of kms to latitude degrees


    ## TODO, fix bounding box
    max_lat = min((lat + latitudeRmax), lat1-resolution)
    min_lat = max((lat - latitudeRmax), lat0)

    minKmPerLonDegree = distance.distance((max_lat, lon), (max_lat, lon+1)).km

    longitudeRmax = rmax/minKmPerLonDegree

    max_lon = min(lon + longitudeRmax, lon1-resolution)
    min_lon = max(lon - longitudeRmax, lon0)

    i0, j0 = getGridCell(min_lat, min_lon, resolution, basin)
    i1, j1 = getGridCell(max_lat, max_lon, resolution, basin)

    touched = []

    for i in range(i0, i1+1):
        for j in range(j0, j1+1):
            if checkCellTouchedByStorm(lat, lon, i, j, rmax, resolution, basin):
                touched.append((i, j))

    return touched

def averageLandfallsPerMonth(TC_data, basin, total_years, resolution, onlyLand=False):
    """
    Calculate average landfalls per month within lat, lon bins from synthetic TC data.
    For use as the training output of ML model.

    :param TC_data: list of synthetically generated TC data
    :param basin: the basin storms are being generated in
    :param total_years: number of years that synthetic data was generated over
    :param resolution: the lat, lon resolution to calculate statistics for in degrees
    :return: a lat, lon, month matrix for the basin with values
             the average number of landfalls in that month in the provided TC data
    """

    # TC_data is list of [year,month,storm_number,l,idx,lat (one),lon (one),pressure (one),wind (one), rmax (one),category,landfall (one),distance]

    # iterate through the data and reorganize into individual storms
    tcs = {}

    grid = createMonthlyLandfallGrid(basin, resolution)

    if len(TC_data) == 0: return grid

    current_storm_id = TC_data[0][2]
    storm = {'t': [], 'data': []}
    for i, storm_point in enumerate(TC_data):

        if i != 0 and storm_point[2] != TC_data[i-1][2]:
            # start a new storm
            tcs[(storm_point[0], storm_point[1], storm_point[2])] = storm.copy()
            storm = {'t': [], 'data': []}

        storm['t'].append(storm_point[3])
        data = storm_point[5:10]
        data.append(storm_point[12])
        storm['data'].append(data)

        if i == len(TC_data) - 1: tcs[(storm_point[0], storm_point[1], storm_point[2])] = storm

    stormsTouchingGridCells = {}  # dict of (lat, lon): { set of storms by (year, month, storm number) }



    for key, tc in tcs.items():
        # interpolate
        b = make_interp_spline(tc['t'], tc['data'], k=1)

        year, month, storm_number = key

        for t in tc['t']:
            step = 1

            # make sure that there is never more than resolution degree change in lat or lon between t values
            if t != len(tc['t']) - 1:

                lat0 = tc['data'][t][0]
                lat1 = tc['data'][t+1][0]
                lon0 = tc['data'][t][1]
                lon1 = tc['data'][t+1][1]


                if abs(lat0 - lat1) + abs(lon0 - lon1) > resolution:
                    # then half the step size until it is small enough

                    n = math.ceil(math.log((abs(tc['data'][t][0] - tc['data'][t+1][0]) + abs(tc['data'][t][1] - tc['data'][t+1][1]))/resolution))

                    step = 1/2**n

            # now use the actual step to check the touching points with spline values
            for j in [h * step + t for h in range(0,int(1/step))]:
                lat, lon, pressure, wind, rmax, dist = b(j)

                if onlyLand and dist > rmax: continue

                touchedCells = getGridCellsTouchedByStormRMax(lat, lon, rmax, resolution, basin)

                for cell in touchedCells:
                    if cell not in stormsTouchingGridCells:
                        stormsTouchingGridCells[cell] = {(year, month, storm_number)}
                    else:
                        stormsTouchingGridCells[cell].add((year, month, storm_number))

    for gridCell, stormsTouchingCell in stormsTouchingGridCells.items():
        latCell, lonCell = gridCell
        for storm in stormsTouchingCell:
            grid[latCell][lonCell][storm[1]-1] += 1

    return grid

