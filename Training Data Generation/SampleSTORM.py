"""
Given STORM model input parameters, generate storm tracks

"""

from STORM.SAMPLE_STARTING_POINT import Startingpoint
from STORM.SAMPLE_TC_MOVEMENT import TC_movement
from STORM.SAMPLE_TC_PRESSURE import TC_pressure
from STORM.SELECT_BASIN import Basins_WMO
import numpy as np
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


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

    loop = 0  # ranges between 0 and 9 to simulate slices of 1000 years

    JM_pressure = np.load(os.path.join(__location__, 'STORM', 'COEFFICIENTS_JM_PRESSURE.npy'), allow_pickle=True).item()

    Genpres = np.load(os.path.join(__location__, 'STORM', 'DP0_PRES_GENESIS.npy'), allow_pickle=True).item()

    # this is the wind pressure relationship coefficients: eq. 3 in the
    WPR_coefficients = np.load(os.path.join(__location__, 'STORM', 'COEFFICIENTS_WPR_PER_MONTH.npy'),
                               allow_pickle=True).item()

    Genwind = np.load(os.path.join(__location__, 'STORM', 'GENESIS_WIND.npy'), allow_pickle=True).item()

    TC_data = []  # This list is composed of: [year, storm number, genesis_month, lat,lon, landfall]
    # ==============================================================================
    #     Step 2: load grid with weighted genesis counts
    # ==============================================================================
    for year in range(0, total_years):
        storms_per_year, genesis_month, lat0, lat1, lon0, lon1 = Basins_WMO(basin)

        if storms_per_year > 0:
            # ==============================================================================
            # Step 3: Generate (list of) genesis locations
            # ==============================================================================
            lon_genesis_list, lat_genesis_list = Startingpoint(storms_per_year, genesis_month, basin, genesis_matrices)

            # ==============================================================================
            # Step 4: Generate initial conditions
            # ==============================================================================
            latlist, lonlist, landfalllist = TC_movement(
                lon_genesis_list,
                lat_genesis_list,
                basin,
                movement_coefficients
            )

            #print(tropical_cyclones[0].latitudes)

            TC_data = TC_pressure(basin,
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

