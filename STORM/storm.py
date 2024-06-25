from TrainingDataGeneration.SampleSTORM import sampleStorm
from TrainingDataGeneration.GenerateInputParameters import getMovementCoefficientData
import numpy as np
import os
from TrainingDataGeneration.RiskFactors import get_landfalls_data
import matplotlib.pyplot as plt

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class STORM:
    monthsall = [
        [6, 7, 8, 9, 10, 11],
        [6, 7, 8, 9, 10, 11],
        [4, 5, 6, 9, 10, 11],
        [1, 2, 3, 4, 11, 12],
        [1, 2, 3, 4, 11, 12],
        [5, 6, 7, 8, 9, 10, 11]
    ]

    def __init__(self, basin="SP", total_years=1000, resolution=0.5, n_years_to_sum=10, n_samples=100):

        self.basin = basin
        self.total_years = total_years
        self.resolution = resolution
        self.n_years_to_sum = n_years_to_sum
        self.n_samples = n_samples

        basins = ["EP", "NA", "NI", "SI", "SP", "WP"]

        monthlist = STORM.monthsall[basins.index(basin)]

        self.month_map = {month: i for i, month in enumerate(STORM.monthsall[basins.index(basin)])}
        print(self.month_map)
        print("Loading files")

        # load all files upfront to store in memory, otherwise parallelization is very slow
        self.JM_pressure = np.load(
            os.path.join(__location__, "COEFFICIENTS_JM_PRESSURE.npy"),
            allow_pickle=True,
        ).item()

        self.JM_pressure_for_basin = np.array(
            [self.JM_pressure[basins.index(basin)][month] for month in monthlist]
        )

        self.Genpres = np.load(
            os.path.join(__location__, "DP0_PRES_GENESIS.npy"), allow_pickle=True
        ).item()
        self.Genpres_for_basin = np.array(
            [self.Genpres[basins.index(basin)][month] for month in monthlist]
        )

        # this is the wind pressure relationship coefficients: eq. 3 in the
        WPR_coefficients = np.load(
            os.path.join(__location__, "COEFFICIENTS_WPR_PER_MONTH.npy"),
            allow_pickle=True,
        ).item()
        self.WPR_coefficients_for_basin = np.array(
            [WPR_coefficients[basins.index(basin)][month] for month in monthlist]
        )

        Genwind = np.load(
            os.path.join(__location__, "GENESIS_WIND.npy"), allow_pickle=True
        ).item()
        self.Genwind_for_basin = [Genwind[basins.index(basin)][month] for month in monthlist]

        self.Penv = {
            month: np.loadtxt(
                os.path.join(
                    __location__, "Monthly_mean_MSLP_" + str(month) + ".txt"
                )
            )
            for month in monthlist
        }

        self.land_mask = np.loadtxt(
            os.path.join(__location__, "Land_ocean_mask_" + str(basin) + ".txt")
        )

        self.mu_list = np.loadtxt(
            os.path.join(__location__, "POISSON_GENESIS_PARAMETERS.txt")
        )

        self.monthlist = np.load(
            os.path.join(__location__, "GENESIS_MONTHS.npy"), allow_pickle=True
        ).item()

        self.rmax_pres = np.load(
            os.path.join(__location__, "RMAX_PRESSURE.npy"), allow_pickle=True
        ).item()

        self.movementCoefficients = getMovementCoefficientData()

    def __call__(self, genesis_matrix):

        storm_arguments = [
                self.JM_pressure_for_basin,
                self.Genpres_for_basin,
                self.WPR_coefficients_for_basin,
                self.Genwind_for_basin,
                self.Penv,
                self.land_mask,
                self.mu_list,
                self.monthlist,
                self.rmax_pres,
                genesis_matrix,
                self.movementCoefficients
            ]

        TC_data = sampleStorm(self.total_years, self.month_map, storm_arguments, on_slurm=False)

        lats = [point[5] for point in TC_data]
        lons = [point[6] for point in TC_data]

        plt.scatter(lats, lons)
        plt.show()

        mean_samples = get_landfalls_data(
            TC_data,
            self.basin,
            self.total_years,
            self.resolution,
            on_slurm=False,
            n_years_to_sum=self.n_years_to_sum,
            n_samples=self.n_samples,
            compute_stats=False
        )

        return np.flipud(np.sum(mean_samples[:,:,:,:3], axis=(-1, -2)))