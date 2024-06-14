"""
    Take a list of storm tracks from storm and compute landfalls per month
"""
import math
import os
import random
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from geopy import distance
from scipy.interpolate import make_interp_spline
from TrainingDataGeneration.STORM.preprocessing import BOUNDARIES_BASINS

def _get_random_year_combination(num_years, size):
    sample = set()
    while len(sample) < size:
        # Choose one random item from items
        elem = random.randint(0, num_years - 1)
        # Using a set elminates duplicates easily
        sample.add(elem)
    return tuple(sample)


def get_random_year_combinations(num_combinations, num_years, size):
    samples = set()
    while len(samples) < num_combinations:
        comb = _get_random_year_combination(num_years, size)
        samples.add(comb)

    return tuple(samples)

def create_monthly_landfall_grid(basin, resolution):
    """
    Generate a grid to store TC monthly landfall statistics
    :param basin
    :param resolution: how many degrees each bin should be
    :return:
    """
    # create lat lon grid (only needs to cover the basin) at 1 degree resolution
    # with [0] * 6 matrix in each

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    shape = (
        math.ceil((lat1 - lat0) * 1 / resolution),
        math.ceil((lon1 - lon0) * 1 / resolution),
        6,  # storm producing months
        6,  # storm categories
    )

    return np.zeros(shape)

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
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    minimum = lat_cell_index * resolution + lat0
    return minimum, minimum + resolution

def get_lon_bounds_for_cell(lon_cell_index, resolution, basin):
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    minimum = lon_cell_index * resolution + lon0
    return minimum, minimum + resolution

def get_closest_lat_in_grid_cell_to_storm_center(lat_cell_index, storm_lat, resolution, basin):
    lat_min, lat_max = get_lat_bounds_for_cell(lat_cell_index, resolution, basin)

    # if the storm is at smaller latitude than the smallest in the cell,
    # the closest we can get is at the smallest latitude in the cell, etc.
    # if the storm is in the lat bounds of the grid cell, we can make the lat distance 0
    if storm_lat < lat_min:
        return lat_min
    if storm_lat > lat_max:
        return lat_max
    return storm_lat

def get_closest_lon_in_grid_cell_to_storm_center(lon_cell_index, storm_lon, resolution, basin):
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
        basin
):
    closest_lat = get_closest_lat_in_grid_cell_to_storm_center(
        lat_cell_index,
        storm_lat,
        resolution,
        basin
    )

    closest_lon = get_closest_lon_in_grid_cell_to_storm_center(
        lon_cell_index,
        storm_lon,
        resolution,
        basin
    )

    storm_distance = distance.distance(
                        (closest_lat, closest_lon),
                        (storm_lat, storm_lon)
                    ).km
    if (storm_distance <= storm_rmax_multiple * storm_radius_max_wind_speeds):
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
    # make bounding box based on storm_radius_max_wind_speeds for which grid cells to iterate through
    lat_min, lat_max, lon_min, lon_max = BOUNDARIES_BASINS(basin)

    storm_rmax_latitude_degrees = (
        storm_rmax_multiple * storm_radius_max_wind_speeds / 111
    )  # approximate conversion of kms to latitude degrees

    storm_max_lat = min((storm_lat + storm_rmax_latitude_degrees), lat_max - resolution)
    storm_min_lat = max((storm_lat - storm_rmax_latitude_degrees), lat_min)

    min_km_per_degree = distance.distance((storm_max_lat, storm_lon), (storm_max_lat, storm_lon + 1)).km

    storm_rmax_longitude_degrees = storm_rmax_multiple * storm_radius_max_wind_speeds / min_km_per_degree

    storm_max_lon = min(storm_lon + storm_rmax_longitude_degrees, lon_max - resolution)
    storm_min_lon = max(storm_lon - storm_rmax_longitude_degrees, lon_min)

    i0, j0 = get_grid_cell(storm_min_lat, storm_min_lon, resolution, basin)
    i1, j1 = get_grid_cell(storm_max_lat, storm_max_lon, resolution, basin)

    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if check_cell_touched_by_storm(
                storm_lat, storm_lon, i, j, storm_radius_max_wind_speeds, storm_rmax_multiple, resolution, basin
            ):
                touched_cells[(i, j)] = max(
                    touched_cells.get((i, j), 0), int(storm_category)
                )

def get_cells_and_sites_touched_by_storm(
    storm, resolution, basin, storm_rmax_multiple
):
    # interpolate
    b = make_interp_spline(storm["t"], storm["data"], k=1)

    touched_cells = {}

    for t in storm["t"]:
        step = 1

        # make sure that there is never more than resolution
        # degree change in lat or lon between t values
        if t != len(storm["t"]) - 1:
            lat0 = storm["data"][t][0]
            lat1 = storm["data"][t + 1][0]
            lon0 = storm["data"][t][1]
            lon1 = storm["data"][t + 1][1]

            if abs(lat0 - lat1) + abs(lon0 - lon1) > resolution:
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
            lat, lon, pressure, wind, storm_radius_max_wind_speeds, category = b(j)

            update_cells_touched_by_storm_rmax(
                touched_cells,
                lat,
                lon,
                storm_radius_max_wind_speeds,
                category,
                storm_rmax_multiple,
                resolution,
                basin
            )

    return touched_cells

def storm_counts_per_month_for_year(
        storms,
        resolution,
        basin,
        storm_rmax_multiple
):

    grid = create_monthly_landfall_grid(basin, resolution)

    month_to_index = {month: i for i, month in enumerate([1, 2, 3, 4, 11, 12])}

    for i, storm in enumerate(storms):
        touched_cells = get_cells_and_sites_touched_by_storm(
            storm,
            resolution,
            basin,
            storm_rmax_multiple=storm_rmax_multiple
        )

        month = storm["month"]

        for cell, category in touched_cells.items():
            lat, lon = cell
            grid[lat][lon][month_to_index[month]][category] += 1

        del touched_cells
    return grid

def get_grid_sum_samples(
    yearly_grids, n_years_to_sum, n_years_to_sum_cat_4_5, n_samples, total_years
):

    sums = []
    yearly_grids = np.array(yearly_grids)
    year_indices = get_random_year_combinations(n_samples, total_years, n_years_to_sum)
    year_indices_cat_4_5 = get_random_year_combinations(
        n_samples, total_years, n_years_to_sum_cat_4_5
    )

    for i in range(n_samples):
        sampled_sum = np.zeros(yearly_grids[0].shape)

        sampled_sum[:, :, :, :4] = np.sum(
            yearly_grids[list(year_indices[i]), :, :, :, :4].copy(), axis=0
        )
        sampled_sum[:, :, :, 4:] = np.sum(
            yearly_grids[list(year_indices_cat_4_5[i]), :, :, :, 4:].copy(), axis=0
        )
        sums.append(sampled_sum)

    sums = np.array(sums)
    return sums

def get_grid_mean_samples(
    yearly_grids, n_years_to_sum, n_years_to_sum_cat_4_5, n_samples, total_years
):
    sums = get_grid_sum_samples(
        yearly_grids, n_years_to_sum, n_years_to_sum_cat_4_5, n_samples, total_years
    )

    return np.mean(sums, axis=0)

def get_landfalls_data(
        TC_data,
        basin,
        total_years,
        resolution,
        storm_rmax_multiple=4,
        local=False
):
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

    print("calculating landfalls")
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

    if local:
        # if not on the cluster you should do this instead
        number_of_cores = cpu_count()
    else:
        # number of cores you have allocated for your slurm task
        number_of_cores = int(os.environ["SLURM_CPUS_PER_TASK"])

    years_of_storms = [[] for i in range(total_years)]

    year = 0

    for storm in storms:
        if storm["year"] >= (year + 1):
            year += 1

        years_of_storms[year].append(storm)

    yearly_grids = []
    with Pool(number_of_cores) as pool:
        args = [
            (year_of_storms, resolution, basin, storm_rmax_multiple)
            for year_of_storms in years_of_storms
        ]

        year_results = pool.starmap(storm_counts_per_month_for_year, args)
        for i, grid in enumerate(year_results):

            yearly_grids.append(grid)

        mean_samples = get_grid_mean_samples(yearly_grids, 10, 100, 100, total_years)
        del yearly_grids

    return mean_samples
