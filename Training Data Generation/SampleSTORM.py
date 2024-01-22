"""
Given STORM model input parameters, generate storm tracks

"""

from STORM.SAMPLE_STARTING_POINT import Startingpoint
from STORM.SAMPLE_TC_MOVEMENT import TC_movement
from STORM.SAMPLE_TC_PRESSURE import TC_pressure
from STORM.SELECT_BASIN import Basins_WMO
import numpy as np
import os
import ray

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def concatenate_ref_results(refs, result=None):

    unfinished = refs
    if result is None:
        result = []

    while unfinished:
        ready_refs, unfinished = ray.wait(unfinished, timeout=None)
        result.extend(ray.get(ready_refs[0]))

    return result


@ray.remote
def stormYear(year, basin, genesis_matrices, movement_coefficients, JM_pressure, Genpres, WPR_coefficients, Genwind):
    storms_per_year, genesis_month, lat0, lat1, lon0, lon1 = Basins_WMO(basin)

    rng = np.random.default_rng()

    TC_data = []

    if storms_per_year > 0:
        # ==============================================================================
        # Step 3: Generate (list of) genesis locations
        # ==============================================================================
        lon_genesis_list, lat_genesis_list = Startingpoint(storms_per_year, genesis_month, basin, genesis_matrices)

        # ==============================================================================
        # Step 4: Generate initial conditions
        # ==============================================================================
        latlist, lonlist, landfalllist = TC_movement(
            rng,
            lon_genesis_list,
            lat_genesis_list,
            basin,
            movement_coefficients
        )

        TC_data = TC_pressure(
            rng,
            basin,
            latlist,
            lonlist,
            landfalllist,
            year,
            storms_per_year,
            genesis_month,
            TC_data,
            JM_pressure,
            Genpres,
            WPR_coefficients,
            Genwind)

    return TC_data


def sampleStorm(total_years, genesis_matrices, movement_coefficients, basin='SP'):
    """
    rewrite of STORM.MASTER to only get TC tracks without intensity data
    :param total_years
    :param genesis_matrices
    :param movement_coefficients
    :param: basin
    :return:
    """

    # ==============================================================================
    # Step 1: Define basin and number of years to run
    # ==============================================================================
    # please set basin (EP,NA,NI,SI,SP,WP)

    JM_pressure = np.load(os.path.join(__location__, 'STORM', 'COEFFICIENTS_JM_PRESSURE.npy'), allow_pickle=True).item()

    Genpres = np.load(os.path.join(__location__, 'STORM', 'DP0_PRES_GENESIS.npy'), allow_pickle=True).item()

    # this is the wind pressure relationship coefficients: eq. 3 in the
    WPR_coefficients = np.load(os.path.join(__location__, 'STORM', 'COEFFICIENTS_WPR_PER_MONTH.npy'),
                               allow_pickle=True).item()

    Genwind = np.load(os.path.join(__location__, 'STORM', 'GENESIS_WIND.npy'), allow_pickle=True).item()

    storm_year_refs = []
    # ==============================================================================
    #     Step 2: load grid with weighted genesis counts
    # ==============================================================================

    # start all year computations as individual workers

    MAX_NUM_PENDING_TASKS = 20

    TC_data = []
    for year in range(0, total_years):
        if len(storm_year_refs) > MAX_NUM_PENDING_TASKS:

            ready_refs, storm_year_refs = ray.wait(storm_year_refs, num_returns=1)

            TC_data.extend(ray.get(ready_refs[0]))

        storm_year_refs.append(
            stormYear.remote(year, basin, genesis_matrices, movement_coefficients, JM_pressure, Genpres,
                             WPR_coefficients, Genwind))

    TC_data = concatenate_ref_results(storm_year_refs, TC_data)

    return TC_data
