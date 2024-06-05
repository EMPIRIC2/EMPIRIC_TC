"""
    Take a list of storm tracks from storm and compute landfalls per month
"""
import math
import os
import random
import time
from multiprocessing import Pool

import numpy as np
from geopy import distance
from scipy.interpolate import make_interp_spline

rmax_multiple = 4

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


month_to_index = {month: i for i, month in enumerate([1, 2, 3, 4, 11, 12])}


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


def createMonthlyLandfallGrid(basin, resolution):
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


def getGridCell(lat, lon, resolution, basin):
    """
    Get the grid cell for given latitude, longitude, and grid resolution

    :return: indices of the lat and lon cells respectively
    """

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    if lat < lat0 or lat >= lat1:
        raise "lat must be within the basin"

    if lon < lon0 or lon >= lon1:
        raise "lon must be within the basin"

    latCell = math.floor((lat - lat0) * 1 / resolution)
    lonCell = math.floor((lon - lon0) * 1 / resolution)

    return latCell, lonCell


def getLatBoundsForCell(latCell, resolution, basin):
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    minimum = latCell * resolution + lat0
    return minimum, minimum + resolution


def getLonBoundsForCell(lonCell, resolution, basin):
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    minimum = lonCell * resolution + lon0
    return minimum, minimum + resolution


def getClosestLatInGridCellToStormCenter(latCell, storm_lat, resolution, basin):
    lat0, lat1 = getLatBoundsForCell(latCell, resolution, basin)

    # if the storm is at smaller latitude than the smallest in the cell,
    # the closest we can get is at the smallest latitude in the cell, etc.
    # if the storm is in the lat bounds of the grid cell, we can make the lat distance 0
    if storm_lat < lat0:
        return lat0
    if storm_lat > lat1:
        return lat1
    return storm_lat


def getClosestLonInGridCellToStormCenter(lonCell, storm_lon, resolution, basin):
    lon0, lon1 = getLonBoundsForCell(lonCell, resolution, basin)

    # same logic as getSmallestLatDistanceInGridCellToStormCenter
    if storm_lon < lon0:
        return lon0
    if storm_lon > lon1:
        return lon1
    return storm_lon


def checkCellTouchedByStorm(
    storm_lat, storm_lon, latCell, lonCell, rmax, rmax_multiple, resolution, basin
):
    closest_lat = getClosestLatInGridCellToStormCenter(
        latCell, storm_lat, resolution, basin
    )
    closest_lon = getClosestLonInGridCellToStormCenter(
        lonCell, storm_lon, resolution, basin
    )

    if (
        distance.distance((closest_lat, closest_lon), (storm_lat, storm_lon)).km
        <= rmax_multiple * rmax
    ):
        return True

    return False


def updateCellsAndSitesTouchedByStormRMax(
    touchedCells,
    lat,
    lon,
    rmax,
    category,
    resolution,
    basin,
    touched_sites,
    sites,
    include_grids,
    include_sites,
):
    # make bounding box based on rmax for which grid cells to iterate through
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    latitudeRmax = (
        rmax_multiple * rmax / 111
    )  # approximate conversion of kms to latitude degrees

    max_lat = min((lat + latitudeRmax), lat1 - resolution)
    min_lat = max((lat - latitudeRmax), lat0)

    minKmPerLonDegree = distance.distance((max_lat, lon), (max_lat, lon + 1)).km

    longitudeRmax = rmax_multiple * rmax / minKmPerLonDegree

    max_lon = min(lon + longitudeRmax, lon1 - resolution)
    min_lon = max(lon - longitudeRmax, lon0)

    if include_grids:
        i0, j0 = getGridCell(min_lat, min_lon, resolution, basin)
        i1, j1 = getGridCell(max_lat, max_lon, resolution, basin)

        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if checkCellTouchedByStorm(
                    lat, lon, i, j, rmax, rmax_multiple, resolution, basin
                ):
                    touchedCells[(i, j)] = max(
                        touchedCells.get((i, j), 0), int(category)
                    )

    if include_sites:
        touched_sites, _ = sites.update_sites_touched_by_storm(
            touched_sites,
            lat,
            lon,
            rmax,
            rmax_multiple,
            (
                (lat - latitudeRmax) - 1,
                (lat + latitudeRmax + 1),
                lon - longitudeRmax - 1,
                lon + longitudeRmax + 1,
            ),
        )
    return (
        (lat - latitudeRmax - 1),
        (lat + latitudeRmax + 1),
        lon - longitudeRmax - 1,
        lon + longitudeRmax + 1,
    )


def get_cells_and_sites_touched_by_storm(
    tc, resolution, basin, sites, include_grids, include_sites
):
    # interpolate
    b = make_interp_spline(tc["t"], tc["data"], k=1)

    touched_cells = {}
    touched_sites = set()

    for t in tc["t"]:
        step = 1

        # make sure that there is never more than resolution
        # degree change in lat or lon between t values
        if t != len(tc["t"]) - 1:
            lat0 = tc["data"][t][0]
            lat1 = tc["data"][t + 1][0]
            lon0 = tc["data"][t][1]
            lon1 = tc["data"][t + 1][1]

            if abs(lat0 - lat1) + abs(lon0 - lon1) > resolution:
                # then half the step size until it is small enough

                n = math.ceil(
                    math.log(
                        (
                            abs(tc["data"][t][0] - tc["data"][t + 1][0])
                            + abs(tc["data"][t][1] - tc["data"][t + 1][1])
                        )
                        / resolution
                    )
                )

                step = 1 / 2**n

        # now use the actual step to check the touching points with spline values
        for j in [h * step + t for h in range(0, int(1 / step))]:
            lat, lon, pressure, wind, rmax, category = b(j)

            updateCellsAndSitesTouchedByStormRMax(
                touched_cells,
                lat,
                lon,
                rmax,
                category,
                resolution,
                basin,
                touched_sites,
                sites,
                include_grids,
                include_sites,
            )

    return touched_cells, touched_sites


def landfallsPerMonthForYear(
    storms, resolution, basin, sites, include_grids, include_sites
):
    print("Starting year", flush=True)
    print(len(storms), flush=True)
    if include_sites:
        site_data = sites.create_site_landfall_vector()
    else:
        site_data = None

    if include_grids:
        grid = createMonthlyLandfallGrid(basin, resolution)
    else:
        grid = None

    for i, storm in enumerate(storms):
        print(i, flush=True)
        touched_cells, touched_sites = get_cells_and_sites_touched_by_storm(
            storm, resolution, basin, sites, include_grids, include_sites
        )

        month = storm["month"]

        if include_grids:
            for cell, category in touched_cells.items():
                lat, lon = cell
                grid[lat][lon][month_to_index[month]][category] += 1

        if include_sites:
            for site in touched_sites:
                site_data[sites.site_to_index[site]][month_to_index[month]][
                    category
                ] += 1
        del touched_cells
        del touched_sites
    print("finish year", flush=True)
    return grid, site_data


def sum_years(yearly_grids, year_index):
    return np.sum(yearly_grids[list(year_index)], axis=0)


def getQuantilesFromYearlyGrids(
    yearly_grids, n_years_to_sum, n_years_to_sum_cat_4_5, n_samples, total_years
):
    print("calculating quantiles")

    year_indices = get_random_year_combinations(n_samples, total_years, n_years_to_sum)
    year_indices_cat_4_5 = get_random_year_combinations(
        n_samples, total_years, n_years_to_sum_cat_4_5
    )

    probs = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    quantiles = []

    yearly_grids = np.array(yearly_grids)

    start = time.time()

    sums = []
    print(yearly_grids.shape)
    for i in range(n_samples):
        cat_0_3 = np.sum(
            yearly_grids[list(year_indices[i]), :, :, :, :4].copy(), axis=0
        )
        cat_4_5 = np.sum(
            yearly_grids[list(year_indices_cat_4_5[i]), :, :, :, 4:].copy(), axis=0
        )

        sums.append(np.concatenate((cat_0_3, cat_4_5), axis=-1))

    end = time.time()

    print("sum time: {}".format(end - start))

    sums = np.array(sums)
    all_categories = np.expand_dims(np.sum(sums, axis=-1), axis=-1)
    sums = np.concatenate((sums, all_categories), axis=-1)

    start = time.time()
    quantiles = np.quantile(sums, probs, axis=0)
    end = time.time()
    print("quantile time: {}".format(end - start))
    return quantiles


def get_grid_sum_samples(
    yearly_grids, n_years_to_sum, n_years_to_sum_cat_4_5, n_samples, total_years
):
    print("getting grid sums", flush=True)

    sums = []
    yearly_grids = np.array(yearly_grids)
    year_indices = get_random_year_combinations(n_samples, total_years, n_years_to_sum)
    year_indices_cat_4_5 = get_random_year_combinations(
        n_samples, total_years, n_years_to_sum_cat_4_5
    )

    for i in range(n_samples):
        print(len(sums))
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


def getLandfallsData(
    TC_data, basin, total_years, resolution, sites, include_grids, include_sites
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
        rmax (one),
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
    print("created storms")

    # number of cores you have allocated for your slurm task
    number_of_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
    # number_of_cores = cpu_count() # if not on the cluster you should do this instead

    years_of_storms = [[] for i in range(total_years)]

    year = 0
    print("n storms", len(storms))
    for storm in storms:
        if storm["year"] >= (year + 1):
            year += 1

        years_of_storms[year].append(storm)

    yearly_grids = []
    yearly_site_data = []
    with Pool(number_of_cores) as pool:
        args = [
            (year_of_storms, resolution, basin, sites, include_grids, include_sites)
            for year_of_storms in years_of_storms
        ]

        year_results = pool.starmap(landfallsPerMonthForYear, args)
        print("computed yearly results", flush=True)
        for i, year_result in enumerate(year_results):
            grid, site_data = year_result

            if include_grids:
                yearly_grids.append(grid)

            if include_sites:
                yearly_site_data.append(site_data)

        mean_samples = get_grid_mean_samples(yearly_grids, 10, 100, 100, total_years)
        del yearly_grids

    return mean_samples, yearly_site_data
