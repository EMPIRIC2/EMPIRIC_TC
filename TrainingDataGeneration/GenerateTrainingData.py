import argparse
import os
import time
import h5py
import numpy as np
from GenerateInputParameters import generateInputParameters, getObservedGenesisLocations
from RiskFactors import getLandfallsData
from SampleSTORM import sampleStorm

from HealthFacilities.getHealthFacilityData import Sites

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

monthsall = [
    [6, 7, 8, 9, 10, 11],
    [6, 7, 8, 9, 10, 11],
    [4, 5, 6, 9, 10, 11],
    [1, 2, 3, 4, 11, 12],
    [1, 2, 3, 4, 11, 12],
    [5, 6, 7, 8, 9, 10, 11],
]
decade_length = 1


def generateOneTrainingDataSample(
        total_years,
        future_data,
        refs,
        sites,
        include_grids,
        include_sites,
        basin="SP",
        include_historical_genesis=False,
        constant_historical_inputs=False,
        compute_stats=False
):
    """
    Generate ML training data

    The ML inputs, X, are randomly generated inputs to the STORM model
    The output targets, Y, are grids of average landfall per month

    :param total_years: number of years to run the STORM simulations over
    :return:
    """

    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]

    monthlist = monthsall[basins.index(basin)]

    (
        genesis_matrices,
        genesis_weightings,
        movement_coefficients,
    ) = generateInputParameters(
        future_data,
        monthlist,
        basin=basin,
        constant_historical_inputs=constant_historical_inputs,
        include_historical_genesis=include_historical_genesis
    )  # replace with generated parameters

    month_map = {key: i for i, key in enumerate(genesis_matrices.keys())}
    genesis_matrix = np.array(
        [np.round(genesis_matrices[month], 1) for month in monthlist]
    )

    refs.append(genesis_matrix)
    refs.append(movement_coefficients)

    tc_data = sampleStorm(total_years, month_map, refs)

    
    outputs = getLandfallsData(
        tc_data, basin, total_years, 0.5, sites, include_grids, include_sites, compute_stats
    )

    if compute_stats:
        grid_means, std_dev, std_devs = outputs
        return genesis_matrix, genesis_weightings, grid_means, std_dev, std_devs, tc_data
    else:
        grid_means, sites = outputs
    # split up input, output data for each month and flatten the matrices
    genesis_matrix = np.nan_to_num(genesis_matrix)

    return genesis_matrix, genesis_weightings, grid_means, sites, tc_data


def generateTrainingData(
    total_years,
    n_train_samples,
    n_test_samples,
    n_validation_samples,
    save_location,
    basin="SP",
    include_grids=False,
    include_sites=False,
    include_historical_genesis=False,
    constant_historical_inputs=False,
    compute_stats=False
):
    print("Beginning TrainingDataGeneration \n")
    print("Running storm for {} years in each sample\n".format(total_years))
    print("Including grid quantiles: {}".format(include_grids))
    print("Training Samples: {}\n".format(n_train_samples))
    print("Test Samples: {}\n".format(n_test_samples))

    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]

    monthlist = monthsall[basins.index(basin)]

    print("Loading files")

    # load all files upfront to store in memory, otherwise parallelization is very slow
    JM_pressure = np.load(
        os.path.join(__location__, "STORM", "COEFFICIENTS_JM_PRESSURE.npy"),
        allow_pickle=True,
    ).item()

    JM_pressure_for_basin = np.array(
        [JM_pressure[basins.index(basin)][month] for month in monthlist]
    )

    Genpres = np.load(
        os.path.join(__location__, "STORM", "DP0_PRES_GENESIS.npy"), allow_pickle=True
    ).item()
    Genpres_for_basin = np.array(
        [Genpres[basins.index(basin)][month] for month in monthlist]
    )

    # this is the wind pressure relationship coefficients: eq. 3 in the
    WPR_coefficients = np.load(
        os.path.join(__location__, "STORM", "COEFFICIENTS_WPR_PER_MONTH.npy"),
        allow_pickle=True,
    ).item()
    WPR_coefficients_for_basin = np.array(
        [WPR_coefficients[basins.index(basin)][month] for month in monthlist]
    )

    Genwind = np.load(
        os.path.join(__location__, "STORM", "GENESIS_WIND.npy"), allow_pickle=True
    ).item()
    Genwind_for_basin = [Genwind[basins.index(basin)][month] for month in monthlist]

    Penv = {
        month: np.loadtxt(
            os.path.join(
                __location__, "STORM", "Monthly_mean_MSLP_" + str(month) + ".txt"
            )
        )
        for month in monthlist
    }

    land_mask = np.loadtxt(
        os.path.join(__location__, "STORM", "Land_ocean_mask_" + str(basin) + ".txt")
    )

    mu_list = np.loadtxt(
        os.path.join(__location__, "STORM", "POISSON_GENESIS_PARAMETERS.txt")
    )

    monthlist = np.load(
        os.path.join(__location__, "STORM", "GENESIS_MONTHS.npy"), allow_pickle=True
    ).item()

    models = ["CMCC-CM2-VHR4", "EC-Earth3P-HR", "CNRM-CM6-1-HR", "HadGEM3-GC31-HM"]
    future_delta_files = [
        os.path.join(
            __location__,
            "InputData",
            "GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model),
        )
        for model in models
    ]

    future_data = [
        np.load(file_path, allow_pickle=True).item()[basin]
        for file_path in future_delta_files
    ]

    sites = Sites(5)
    print("n sites", len(sites.sites))

    file_time = time.time()
    with h5py.File(
        os.path.join(save_location, "AllData_{}.hdf5".format(file_time)), "w-"
    ) as data:
        lat, lon = future_data[0][1].shape


        if include_historical_genesis:
            n_weights = 5
        else:
            n_weights = 4

        if not constant_historical_inputs:
            data.create_dataset("train_genesis_weightings", (n_train_samples, n_weights))
            data.create_dataset("test_genesis_weightings", (n_test_samples, n_weights))
            data.create_dataset("validation_genesis_weightings", (n_train_samples, n_weights))

        data.create_dataset("train_genesis", (n_train_samples, 6, lat, lon))
        data.create_dataset("test_genesis", (n_test_samples, 6, lat, lon))
        data.create_dataset("validation_genesis", (n_validation_samples, 6, lat, lon))

        if include_grids:
            
            if compute_stats:
                shape = (2 * lat, 2 * lon)
                data.create_dataset(
                    "train_stds", (n_train_samples, *shape)
                )
                data.create_dataset(
                    "validation_stds", (n_validation_samples, *shape)
                )
                data.create_dataset(
                    "test_stds", (n_test_samples, *shape)
                )
            else:
                shape = (2 * lat, 2 * lon, 6, 6)
             
            data.create_dataset(
                "train_grids", (n_train_samples, *shape)
            )
            data.create_dataset("test_grids", (n_test_samples, *shape))
            data.create_dataset(
                "validation_grids", (n_validation_samples, *shape)
            )
            
        if include_sites:
            data.create_dataset(
                "train_sites", (n_train_samples, total_years, 530, 6, 5)
            )
            data.create_dataset("test_sites", (n_test_samples, total_years, 530, 6, 5))
            data.create_dataset(
                "validation_sites", (n_validation_samples, total_years, 530, 6, 5)
            )

    rmax_pres = np.load(
        os.path.join(__location__, "STORM", "RMAX_PRESSURE.npy"), allow_pickle=True
    ).item()

    print("Finished loading files")

    all_tc_data = []

    print("Generating samples")
    dataset = "train"
    offset = 0
    for i in range(n_train_samples + n_test_samples + n_validation_samples):
        if i == n_train_samples:
            dataset = "test"
            offset = n_train_samples
            print("Generating test samples")

        if i == n_train_samples + n_test_samples:
            dataset = "validation"
            offset = n_train_samples + n_test_samples
            print("Generating validation samples")

        print("Generating {} sample: {}".format(dataset, i - offset))

        outputs = generateOneTrainingDataSample(
            total_years,
            future_data,
            refs=
            [
                JM_pressure_for_basin,
                Genpres_for_basin,
                WPR_coefficients_for_basin,
                Genwind_for_basin,
                Penv,
                land_mask,
                mu_list,
                monthlist,
                rmax_pres,
            ],
            sites=sites,
            include_grids=include_grids,
            include_sites=include_sites,
            basin=basin,
            constant_historical_inputs=constant_historical_inputs,
            include_historical_genesis=include_historical_genesis,
            compute_stats=compute_stats
        )

        if compute_stats:
            (  genesis_matrices,
                genesis_weightings,
                grid_means,
                grid_std,
                stds,
                tc_data
            ) = outputs
        else:
            (
                genesis_matrices,
                genesis_weightings,
                grid_means,
                yearly_site_data,
                tc_data,
            ) = outputs
            
        with h5py.File(
            os.path.join(save_location, "AllData_{}.hdf5".format(file_time)), "r+"
        ) as data:
            data["{}_genesis".format(dataset)][i - offset] = genesis_matrices

            if not constant_historical_inputs:
                data["{}_genesis_weightings".format(dataset)][
                    i - offset
                ] = genesis_weightings

            if include_grids:
                if compute_stats:
                    data["{}_stds".format(dataset)][i-offset] = grid_std
                    
                data["{}_grids".format(dataset)][i - offset] = grid_means
           
            if include_sites:
                data["{}_sites".format(dataset)][i - offset] = yearly_site_data
        #all_tc_data.append(tc_data)
        
        if compute_stats:
            np.save(os.path.join(save_location, "stds_{}_{}".format(file_time, i)),
                np.array(decadal_stds, dtype=object),
                allow_pickle=True,
            )

    if save_location is not None:
        print("Saving to: {}".format(save_location))

        np.save(
            os.path.join(save_location, "tc_data_{}".format(file_time)),
            np.array(all_tc_data, dtype=object),
            allow_pickle=True,
        )

    print("TrainingDataGeneration Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate machine learning training data from STORM"
    )
    parser.add_argument(
        "total_years",
        type=int,
        help="Number of years to run STORM for when generating training data",
    )
    parser.add_argument("num_training", type=int, help="Number of training samples")
    parser.add_argument("num_test", type=int, help="number of test samples")
    parser.add_argument("num_validation", type=int, help="number of validation samples")
    parser.add_argument("save_location", type=str, help="Directory to save data to.")
    parser.add_argument(
        "--include_grids",
        action="store_true",
        help="If True generate gridded quantile data",
        default=False,
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help="If True compute std of decade counts",
        default=False
    )

    parser.add_argument(
        "--include_sites",
        action="store_true",
        help="If True generate site specific data",
        default=False,
    )

    parser.add_argument(
        "--include_historical_genesis",
        action="store_true",
        help="If True include the historical genesis " 
             "data when sampling random input maps",
        default=False
    )

    parser.add_argument(
        "--constant_historical_inputs",
        action="store_true",
        help="If True do not sample random input maps and only use the historical",
        default=False
    )

    args = parser.parse_args()
    if not (args.include_grids or args.include_sites):
        raise Exception("ERROR: Must generate either grids or sites as the output data")

    if (args.include_historical_genesis and args.constant_historical_inputs):
        raise Exception("ERROR: --include_historical_genesis and "
                        "--constant_historical_inputs cannot both be set")

    generateTrainingData(
        args.total_years,
        args.num_training,
        args.num_test,
        args.num_validation,
        args.save_location,
        include_grids=args.include_grids,
        include_sites=args.include_sites,
        include_historical_genesis=args.include_historical_genesis,
        constant_historical_inputs=args.constant_historical_inputs,
        compute_stats=args.compute_stats
    )
