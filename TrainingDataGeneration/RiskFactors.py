"""
    Take a list of storm tracks from storm and compute landfalls per month
"""
import math
import os
import random
from math import asin, cos, radians, sin, sqrt
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.interpolate import make_interp_spline

from STORM.preprocessing import BOUNDARIES_BASINS, find_basin



def _get_random_year_combination(num_years, size):
    sample = set()
    while len(sample) < size:
        year = random.randint(0, num_years - 1)
        sample.add(year)
    return tuple(sample)


def get_random_year_combinations(num_combinations, num_years, size):
    
    if math.comb(num_years, size) < num_combinations: raise Exception("Not enough years to make {} combinations".format(num_combinations))
    
    samples = set()
    while len(samples) < num_combinations:
        comb = _get_random_year_combination(num_years, size)
        samples.add(comb)

    return tuple(samples)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    taken from https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


month_to_index = {month: i for i, month in enumerate([1, 2, 3, 4, 11, 12])}


def create_monthly_landfall_grid(basin: str, resolution=0.5):
    """
    Generate a grid to store TC monthly landfall statistics
    :param basin: which ocean basin to create the grid for
    :param resolution: how many degrees each bin should be
    :return: a 4d ndarray
    """

    lat_min, lat_max, lon_min, lon_max = BOUNDARIES_BASINS(basin)

    shape = (
        math.ceil((lat_max - lat_min) * 1 / resolution),
        math.ceil((lon_max - lon_min) * 1 / resolution),
        6,  # storm producing months
        6,  # storm categories
    )

    return np.zeros(shape, dtype=np.uint8)


def get_grid_cell(lat, lon, resolution, basin):
    """
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    """

    lat_min, lat_max, lon_min, lon_max = BOUNDARIES_BASINS(basin)

    if not lat_min <= lat < lat_max:
        raise "lat must be within the basin"

    if not lon_min <= lon < lon_max:
        raise "lon must be within the basin"

    lat_cell_index = math.floor((lat - lat_min) * 1 / resolution)
    lon_cell_index = math.floor((lon - lon_min) * 1 / resolution)

    return lat_cell_index, lon_cell_index


def get_lat_bounds_for_cell(lat_cell_index, resolution, basin):
    lat_min, lat_max, lon_min, lon_max = BOUNDARIES_BASINS(basin)

    minimum = lat_cell_index * resolution + lat_min
    return minimum, minimum + resolution


def get_lon_bounds_for_cell(lon_cell_index, resolution, basin):
    lat_min, lat_max, lon_min, lon_max = BOUNDARIES_BASINS(basin)

    minimum = lon_cell_index * resolution + lon_min
    return minimum, minimum + resolution


def get_closest_lat_in_grid_cell_to_storm_center(
    lat_cell_index, storm_lat, resolution, basin
):
    lat_min, lat_max = get_lat_bounds_for_cell(lat_cell_index, resolution, basin)

    # if the storm is at smaller latitude than the smallest in the cell,
    # the closest we can get is at the smallest latitude in the cell, etc.
    # if the storm is in the lat bounds of the grid cell, we can make the lat distance 0
    if storm_lat < lat_min:
        return lat_min
    if storm_lat > lat_max:
        return lat_max
    return storm_lat


def get_closest_lon_in_grid_cell_to_storm_center(
    lon_cell_index, storm_lon, resolution, basin
):
    lon_min, lon_max = get_lon_bounds_for_cell(lon_cell_index, resolution, basin)

    # same logic as getSmallestLatDistanceInGridCellToStormCenter
    if storm_lon < lon_min:
        return lon_min
    if storm_lon > lon_max:
        return lon_max
    return storm_lon


def check_cell_touched_by_storm(
    storm_lat,
    storm_lon,
    lat_cell_index,
    lon_cell_index,
    storm_radius_max_wind_speeds,
    storm_rmax_multiple,
    resolution,
    basin,
):
    closest_lat = get_closest_lat_in_grid_cell_to_storm_center(
        lat_cell_index, storm_lat, resolution, basin
    )

    closest_lon = get_closest_lon_in_grid_cell_to_storm_center(
        lon_cell_index, storm_lon, resolution, basin
    )

    if (
        haversine(closest_lat, closest_lon,storm_lat, storm_lon)

        <= storm_rmax_multiple * storm_radius_max_wind_speeds
    ):
        return True

    return False


def update_cells_touched_by_storm_rmax(
    touched_cells,
    storm_lat,
    storm_lon,
    storm_radius_max_wind_speeds,
    storm_category,
    storm_rmax_multiple,
    resolution,
    basin,
):
    """

    @param touched_cells:
    @param storm_lat:
    @param storm_lon:
    @param storm_radius_max_wind_speeds:
    @param storm_category:
    @param storm_rmax_multiple:
    @param resolution:
    @param basin:
    @return:
    """

    # make bounding box based on storm_radius_max_wind_speeds
    # for which grid cells to check
    lat_min, lat_max, lon_min, lon_max = BOUNDARIES_BASINS(basin)

    storm_rmax_latitude_degrees = (
        storm_rmax_multiple * storm_radius_max_wind_speeds / 111
    )  # approximate conversion of kms to latitude degrees

    storm_max_lat = min((storm_lat + storm_rmax_latitude_degrees), lat_max - resolution)
    storm_min_lat = max((storm_lat - storm_rmax_latitude_degrees), lat_min)

    min_km_per_degree = haversine(
        storm_max_lat, storm_lon, storm_max_lat, storm_lon + 1
    )

    storm_rmax_longitude_degrees = (
        storm_rmax_multiple * storm_radius_max_wind_speeds / min_km_per_degree
    )

    storm_max_lon = min(storm_lon + storm_rmax_longitude_degrees, lon_max - resolution)
    storm_min_lon = max(storm_lon - storm_rmax_longitude_degrees, lon_min)

    lat_index_min, lon_index_min = get_grid_cell(
        storm_min_lat, storm_min_lon, resolution, basin
    )

    lat_index_max, lon_index_max = get_grid_cell(
        storm_max_lat, storm_max_lon, resolution, basin
    )
    for lat_index in range(lat_index_min, lat_index_max + 1):
        for lon_index in range(lon_index_min, lon_index_max + 1):
            if check_cell_touched_by_storm(
                storm_lat,
                storm_lon,
                lat_index,
                lon_index,
                storm_radius_max_wind_speeds,
                storm_rmax_multiple,
                resolution,
                basin,
            ):
                current_category = touched_cells.get((lat_index, lon_index), 0)
                touched_cells[(lat_index, lon_index)] = max(
                    current_category, int(storm_category)
                )

def get_cells_and_sites_touched_by_storm(storm, resolution, basin, storm_rmax_multiple):
    # interpolate
    bspline = make_interp_spline(storm["t"], storm["data"], k=1)

    touched_cells = {}

    for t in storm["t"]:
        step = 1

        # make sure that there is never more than resolution
        # degree change in lat or lon between t values
        if t != len(storm["t"]) - 1:
            lat_min = storm["data"][t][0]
            lat_max = storm["data"][t + 1][0]
            lon_min = storm["data"][t][1]
            lon_max = storm["data"][t + 1][1]

            if abs(lat_min - lat_max) + abs(lon_min - lon_max) > resolution:
                # then half the step size until it is small enough

                n = math.ceil(
                    math.log(
                        (
                            abs(storm["data"][t][0] - storm["data"][t + 1][0])
                            + abs(storm["data"][t][1] - storm["data"][t + 1][1])
                        )
                        / resolution
                    )
                )

                step = 1 / 2**n

        # now use the actual step to check the touching points with spline values
        for j in [h * step + t for h in range(0, int(1 / step))]:
            lat, lon, pressure, wind, storm_radius_max_wind_speeds, category = bspline(
                j
            )

            update_cells_touched_by_storm_rmax(
                touched_cells,
                lat,
                lon,
                storm_radius_max_wind_speeds,
                category,
                storm_rmax_multiple,
                resolution,
                basin,
            )

    return touched_cells


def storm_counts_per_month(storms, resolution, basin, storm_rmax_multiple):
    """
    Takes storms and computes the storm counts on a lat lon grid,
     separated by month and category
    @param storms: a list of storms
    a storm has format {"month": int, "year": int, "t": [0,..., n], "data": [...]}
    @param resolution: size of lat/lon degree cells in degrees to compute counts in
    @param basin: the ocean basin
    @param storm_rmax_multiple: multiple of storm radius of maximum
    wind speeds that determines whether a storm touches a cell.
    @return: returns a 4d np.ndarray with dimensions
    (lat_cells, lon_cells, months, category)
    """

    grid = create_monthly_landfall_grid(basin, resolution)

    month_to_index = {month: i for i, month in enumerate([1, 2, 3, 4, 11, 12])}

    for i, storm in enumerate(storms):
        touched_cells = get_cells_and_sites_touched_by_storm(
            storm, resolution, basin, storm_rmax_multiple=storm_rmax_multiple
        )
        month = storm["month"]
        for cell, category in touched_cells.items():
            lat, lon = cell
            grid[lat][lon][month_to_index[month]][category] += 1

        del touched_cells
    return grid

def get_grid_sum_samples(
    yearly_grids, n_years_to_sum, n_samples, total_years
):
    sums = []
    yearly_grids = np.array(yearly_grids)
    year_indices = get_random_year_combinations(n_samples, total_years, n_years_to_sum)

    for i in range(n_samples):
        sampled_sum = np.zeros(yearly_grids[0].shape)

        sampled_sum[:, :, :, :] = np.sum(
            yearly_grids[list(year_indices[i]), :, :, :, :].copy(), axis=0
        )
        sums.append(sampled_sum)

    sums = np.array(sums)
    return sums


def get_grid_decade_statistics(
    yearly_grids, n_years_to_sum, n_samples, total_years
):
    yearly_grids = np.array(yearly_grids)
    year_indices = get_random_year_combinations(n_samples, total_years, n_years_to_sum)


    s_1 = np.zeros(shape=yearly_grids.shape[1:3])
    s_2 = np.zeros(shape=yearly_grids.shape[1:3])
    std_devs = []
    for i in range(n_samples):

        sampled_sum = np.zeros(yearly_grids[0].shape)

        sampled_sum[:, :, :, :] = np.sum(
            yearly_grids[list(year_indices[i]), :, :, :, :].copy(), axis=0
        )

        s_1 += np.sum(sampled_sum[:, :, :, :], axis=(-1, -2))
        s_2 += np.sum(sampled_sum[:, :, :, :], axis=(-1, -2)) ** 2
        std_dev = np.sqrt(s_2 / (i+1) - (s_1 / (i+1)) ** 2)
        std_devs.append(np.max(std_dev))
        del sampled_sum

    mean = s_1 / n_samples
    return mean, std_dev, std_devs


def get_grid_mean_samples(
    yearly_grids, n_years_to_sum, n_samples, total_years
):
    sums = get_grid_sum_samples(
        yearly_grids, n_years_to_sum, n_samples, total_years
    )

    return np.mean(sums, axis=0)



def get_landfalls_data(
        TC_data,
        basin,
        total_years,
        resolution,
        on_slurm,
        storm_rmax_multiple=4
):
    """
    Take tropical cyclone tracks and data from STORM and calculate mean
    tropical cyclone counts in lat, lon bins per month and storm category
    For use as the training output of an ML model.

    @param TC_data: list of synthetically generated TC data
    @param basin: the basin storms are being generated in
    @param total_years: number of years that synthetic data was generated over
    @param resolution: the lat, lon resolution to calculate statistics for in degrees
    @param storm_rmax_multiple: constant that storm_rmax is multiplied by to
    determine if it touches a region
    @param on_slurm: boolean flag, true if code is being run as a slurm job
    @return: a lat, lon, month matrix for the basin with values
             the average number of landfalls in that month in the provided TC data
    """

    """
    TC_data is list of [
        year,
        month,
        storm_number,
        l,
        idx,
        lat (one step),
        lon (one),
        pressure (one),
        wind (one),
        storm_radius_max_wind_speeds (one),
        category,
        landfall (one),
        distance
    ]
    """

    storms = []

    storm = {"t": [], "data": []}
    for i, storm_point in enumerate(TC_data):
        if i != 0 and storm_point[2] != TC_data[i - 1][2]:
            storm["month"] = storm_point[1]
            storm["year"] = storm_point[0]
            storm["t"] = [i for i in range(len(storm["data"]))]
            storms.append(storm)

            storm = {"t": [], "data": [], "month": None}

        data = storm_point[5:11]
        storm["data"].append(data)

        if i == len(TC_data) - 1:
            storm["month"] = storm_point[1]
            storm["t"] = [i for i in range(len(storm["data"]))]
            storm["year"] = storm_point[0]
            storms.append(storm)

    if on_slurm:
        # number of cores you have allocated for your slurm task
        number_of_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        # if not on the cluster you should do this instead
        number_of_cores = cpu_count()

    decades_of_storms = [[] for i in range(total_years)]

    decade = 0

    for storm in storms:
        if storm["year"] >= (decade*10 + 10):
            decade += 1

        decades_of_storms[decade].append(storm)

    with Pool(number_of_cores) as pool:
        args = [
            (decade_of_storms, resolution, basin, storm_rmax_multiple)
            for decade_of_storms in decades_of_storms
        ]

        decade_results = pool.starmap(storm_counts_per_month, args)

        means = []
        std_devs = []

        if total_years >= 1000:
            for i in range(1, total_years // 1000 + 1):
                means.append(np.mean(decade_results[:100 * i], axis=0))
                std_devs.append(np.std(decade_results[:100 * i], axis=0))

    return np.array(means), np.array(std_devs)
