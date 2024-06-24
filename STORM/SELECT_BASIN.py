# -*- coding: utf-8 -*-
"""
@author: Nadia Bloemendaal, nadia.bloemendaal@vu.nl

For more information, please see
Bloemendaal, N., Haigh, I.D., de Moel, H. et al.
Generation of a global synthetic tropical cyclone hazard dataset using STORM.
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

This is the STORM module for simulation of genesis month, frequency, and basin boundaries

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""
import os
import random
import sys

import numpy as np

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Basin indices:
# 0 = EP = Eastern Pacific
# 1 = NA = North Atlantic
# 2 = NI = North Indian
# 3 = SI = South Indian
# 4 = SP = South Pacific
# 5 = WP = Western Pacific


def Genesis_month(idx, storms, monthlist):
    """
    Sample the genesis months for every TC
    Parameters
    ----------
    idx : basin index (0=EP 1=NA 2=NI 3=SI 4=SP 5=WP).
    storms : number of TCs.

    Returns
    -------
    monthall : list of all genesis months.

    """

    monthall = []
    for i in range(0, storms):
        monthall.append(np.random.choice(monthlist[idx]))

    return monthall


def Storms(idx, mu_list):
    """
    Sample the number of TC formations in a given year

    Parameters
    ----------
    idx : basin index (0=EP 1=NA 2=NI 3=SI 4=SP 5=WP).

    Returns
    -------
    s : number of storms.

    """
    # mu_list has the shape [EP,NA,NI,SI,SP,WP]

    mu = mu_list[idx]
    s=np.random.poisson(mu,1)[0]
    return s

def get_basin_boundaries(basin):

    if basin == "EP":  # Eastern Pacific
        lat_min, lat_max, lon_min, lon_max = 5, 60, 180, 285
    elif basin == "NA":  # North Atlantic
        lat_min, lat_max, lon_min, lon_max = 5, 60, 255, 359
    elif basin == "NI":  # North Indian
        lat_min, lat_max, lon_min, lon_max = 5, 60, 30, 100
    elif basin == "SI":  # South Indian
        lat_min, lat_max, lon_min, lon_max = -60, -5, 10, 135
    elif basin == "SP":  # South Pacific
        lat_min, lat_max, lon_min, lon_max = -60, -5, 135, 240
    elif basin == "WP":  # Western Pacific
        lat_min, lat_max, lon_min, lon_max = 5, 60, 100, 180
    else:
        raise Exception("Invalid Basin")

    return lat_min, lat_max, lon_min, lon_max
def Basins_WMO(basin, mu_list, monthlist):
    """
    Basin definitions

    Parameters
    ----------
    basin : basin.

    Returns
    -------
    s : number of storms.
    month : list of genesis months.
    lat_min : lower left corner latitude.
    lat_max : upper right corner latitude.
    lon_min : lower left corner longitude.
    lon_max : upper right corner longitude.

    """
    # We follow the basin definitions from the IBTrACS dataset, but with lat boundaries set at 60 N/S
    # The ENP/AO border will be defined in the algorithm later.

    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]
    basin_name = dict(zip(basins, [0, 1, 2, 3, 4, 5]))
    idx = basin_name[basin]

    lat_min, lat_max, lon_min, lon_max = get_basin_boundaries(basin)

    s = Storms(idx, mu_list)

    month = Genesis_month(idx, s, monthlist)

    return s, month, lat_min, lat_max, lon_min, lon_max
