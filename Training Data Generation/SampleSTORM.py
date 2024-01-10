"""
Given STORM model input parameters, generate storm tracks

"""

from STORM.SAMPLE_STARTING_POINT import Startingpoint
from STORM.SAMPLE_TC_MOVEMENT import TC_movement
from STORM.SELECT_BASIN import Basins_WMO
import numpy as np

def sampleStorm(total_years, genesis_matrices, movement_coefficients, basin='EP'):
    """
    rewrite of STORM.MASTER to only get TC tracks without intensity data

    :param basin
    :param: total_years
    :return:
    """

    # ==============================================================================
    # Step 1: Define basin and number of years to run
    # ==============================================================================
    # please set basin (EP,NA,NI,SI,SP,WP)

    loop = 0  # ranges between 0 and 9 to simulate slices of 1000 years

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
            latlist, lonlist, landfalllist = TC_movement(lon_genesis_list, lat_genesis_list, basin, movement_coefficients)
            print(lonlist)
            TC_data += [[year, storm_number, genesis_month[storm_number], latlist[storm_number], lonlist[storm_number], landfalllist] for storm_number in range(storms_per_year)]

    return TC_data

#sampleStorm(1)