"""
Given STORM model input parameters, generate storm tracks

"""

from STORM.SAMPLE_STARTING_POINT import Startingpoint
from STORM.SAMPLE_TC_MOVEMENT import TC_movement
from STORM.SAMPLE_TC_PRESSURE import TC_pressure
from STORM.SELECT_BASIN import Basins_WMO
import numpy as np
from multiprocessing import Pool, cpu_count


import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))



def stormYear(year, month_map, basin, JM_pressure, Genpres, WPR_coefficients, Genwind, Penv, land_mask, mu_list, monthlist, rmax_pres, genesis_matrix, movement_coefficients):

    storms_per_year, genesis_month, lat0, lat1, lon0, lon1 = Basins_WMO(basin, mu_list, monthlist)


    rng = np.random.default_rng()

    TC_data = []

    if storms_per_year > 0:
        # ==============================================================================
        # Step 3: Generate (list of) genesis locations
        # ==============================================================================
        lon_genesis_list, lat_genesis_list = Startingpoint(storms_per_year, genesis_month, basin, genesis_matrix, month_map, land_mask, mu_list, monthlist)

        # ==============================================================================
        # Step 4: Generate initial conditions
        # ==============================================================================
        latlist, lonlist, landfalllist = TC_movement(
            rng,
            lon_genesis_list,
            lat_genesis_list,
            basin,
            movement_coefficients,
            mu_list,
            monthlist,
            land_mask
        )

        TC_data = TC_pressure(
            rng,
            month_map,
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
            Genwind,
            Penv,
            mu_list,
            monthlist,
            rmax_pres
        )

    return TC_data


def sampleStorm(total_years,
                month_map,
                refs,
                basin='SP'):
    """
    rewrite of STORM.MASTER to only get TC tracks without intensity data
    :param total_years
    :param genesis_matrices
    :param movement_coefficients
    :param: basin
    :return:
    """

    # number of cores you have allocated for your slurm task:
    #number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    number_of_cores = cpu_count() # if not on the cluster you should do this instead
    
    args = [(year, month_map, basin, *refs) for year in range(total_years)]
    # multiprocssing pool to distribute tasks to:
    with Pool(number_of_cores) as pool:
        # distribute computations and collect results:
        results = pool.starmap(stormYear, args)

    TC_data = []
    for result in results:
        TC_data += result

    return TC_data
