from SampleSTORM import sampleStorm
from RiskFactors import getLandfallsData
from GenerateInputParameters import generateInputParameters
from getHealthFacilityData import Sites
import os
import numpy as np
import argparse
import h5py
import time
from PlotMapData import plotLatLonGridData
import matplotlib.pyplot as plt

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

monthsall=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
decade_length = 1


def generateOneTrainingDataSample(total_years, future_data, movementCoefficientsFuture, refs, sites, basin='SP', include_grids=False):
    '''
    Generate ML training data

    The ML inputs, X, are randomly generated inputs to the STORM model
    The output targets, Y, are grids of average landfall per month

    :param total_years: number of years to run the STORM simulations over
    :return:
    '''

    basins = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']

    monthlist = monthsall[basins.index(basin)]

    genesis_matrices, movement_coefficients = generateInputParameters(future_data, movementCoefficientsFuture, monthlist) # replace with generated parameters

    month_map = {key: i for i, key in enumerate(genesis_matrices.keys())}
    genesis_matrix = np.array([np.round(genesis_matrices[month],1) for month in monthlist])

    refs.append(genesis_matrix)
    refs.append(movement_coefficients)

    tc_data = sampleStorm(total_years, month_map, refs)

    decade_grids, decade_site_data = getLandfallsData(tc_data, basin, total_years, .5, sites, include_grids)
    basin_movement_coefficients = movement_coefficients[basins.index(basin)]

    # split up input, output data for each month and flatten the matrices
    genesis_matrix = np.nan_to_num(genesis_matrix)

    return (genesis_matrix, basin_movement_coefficients), decade_site_data, tc_data

def generateTrainingData(total_years, n_train_samples, n_test_samples, n_validation_samples, save_location, basin='SP'):

    print('Beginning Training Data Generation \n')
    print('Running storm for {} years in each sample\n'.format(total_years))
    print('Training Samples: {}\n'.format(n_train_samples))
    print('Test Samples: {}\n'.format(n_test_samples))

    basins = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']

    monthlist = monthsall[basins.index(basin)]


    all_train_inputs = []
    all_train_outputs = []

    all_test_inputs = []
    all_test_outputs = []

    training_sample_refs = []
    MAX_NUM_PENDING_TASKS = 20

    print("Loading files")
    ## load all files upfront to store in memory, otherwise parallelization is very slow
    JM_pressure = np.load(os.path.join(__location__, 'STORM', 'COEFFICIENTS_JM_PRESSURE.npy'), allow_pickle=True).item()

    JM_pressure_for_basin = np.array([JM_pressure[basins.index(basin)][month] for month in monthlist])


    Genpres = np.load(os.path.join(__location__, 'STORM', 'DP0_PRES_GENESIS.npy'), allow_pickle=True).item()
    Genpres_for_basin = np.array([Genpres[basins.index(basin)][month] for month in monthlist])

    # this is the wind pressure relationship coefficients: eq. 3 in the
    WPR_coefficients = np.load(os.path.join(__location__, 'STORM', 'COEFFICIENTS_WPR_PER_MONTH.npy'),
                               allow_pickle=True).item()
    WPR_coefficients_for_basin = np.array([WPR_coefficients[basins.index(basin)][month] for month in monthlist])


    Genwind = np.load(os.path.join(__location__, 'STORM', 'GENESIS_WIND.npy'), allow_pickle=True).item()
    Genwind_for_basin = [Genwind[basins.index(basin)][month] for month in monthlist]

    Penv = {month: np.loadtxt(os.path.join(__location__, 'STORM', 'Monthly_mean_MSLP_' + str(month) + '.txt')) for month in monthlist}

    land_mask = np.loadtxt(os.path.join(__location__, 'STORM', 'Land_ocean_mask_'+str(basin)+'.txt'))

    mu_list = np.loadtxt(os.path.join(__location__,'STORM', 'POISSON_GENESIS_PARAMETERS.txt'))

    monthlist=np.load(os.path.join(__location__,'STORM','GENESIS_MONTHS.npy'), allow_pickle=True).item()

    movementCoefficientsFuture = [np.load(os.path.join(__location__, 'InputData', "JM_LONLATBINS_IBTRACSDELTA_{}.npy".format(model))
                                          , allow_pickle=True)
                                  .item()['SP']
                                  for model in ['CMCC-CM2-VHR4','EC-Earth3P-HR','CNRM-CM6-1-HR','HadGEM3-GC31-HM']]

    models = ['CMCC-CM2-VHR4', 'EC-Earth3P-HR', 'CNRM-CM6-1-HR', 'HadGEM3-GC31-HM']
    future_delta_files = [os.path.join(__location__, 'InputData', "GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model))
                          for model in models]

    future_data = [np.load(file_path, allow_pickle=True).item()[basin] for file_path in future_delta_files]
    site_files = [os.path.join(__location__, file_name) for file_name in ['SPC_health_data_hub_Kiribati.csv', 'SPC_health_data_hub_Solomon_Islands.csv', 'SPC_health_data_hub_Tonga.csv', 'SPC_health_data_hub_Vanuatu.csv']]
    sites = Sites(site_files, 5)


    file_time = time.time()
    with h5py.File(os.path.join(save_location, 'AllData_{}.hdf5'.format(file_time)), 'w-') as data:

        lat, lon = future_data[0][1].shape

        data.create_dataset('train_genesis', (n_train_samples, 6, lat, lon))
        data.create_dataset('test_genesis', (n_test_samples, 6, lat, lon))
        data.create_dataset('validation_genesis', (n_validation_samples, 6, lat, lon))

        w = len(movementCoefficientsFuture[0])
        h = len(movementCoefficientsFuture[0][0])
        data.create_dataset('train_movement', (n_train_samples, w, h))
        data.create_dataset('test_movement', (n_test_samples, w, h))
        data.create_dataset('validation_movement', (n_validation_samples, w, h))

        data.create_dataset('train_grids', (n_train_samples, total_years, 2*lat, 2*lon, 6, 5))
        data.create_dataset('test_grids', (n_test_samples, total_years, 2*lat, 2*lon, 6, 5))
        data.create_dataset('train_sites', (n_train_samples, total_years, len(sites.sites), 6, 5))
        data.create_dataset('test_sites', (n_test_samples, total_years, len(sites.sites), 6, 5))

        data.create_dataset('validation_grids', (n_validation_samples, total_years, 2*lat, 2*lon, 6, 5))
        data.create_dataset('validation_sites', (n_validation_samples, total_years, len(sites.sites), 6, 5))

    rmax_pres = np.load(os.path.join(__location__, 'STORM', 'RMAX_PRESSURE.npy'),allow_pickle=True).item()

    print("Finished loading files")

    all_train_tc_data = []
    all_test_tc_data = []

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

        input, decade_site_data,  tc_data = generateOneTrainingDataSample(
            total_years,
            future_data,
            movementCoefficientsFuture,
            [JM_pressure_for_basin,
                Genpres_for_basin,
                WPR_coefficients_for_basin,
                Genwind_for_basin,
                Penv,
                land_mask,
                mu_list,
                monthlist,
                rmax_pres],
            sites,
            basin
        )

        with h5py.File(os.path.join(save_location, 'AllData_{}.hdf5'.format(file_time)), 'r+') as data:
            genesis_matrices, movement_coefficients = input

            data['{}_genesis'.format(dataset)][i - offset] = genesis_matrices
            data['{}_movement'.format(dataset)][i - offset] = movement_coefficients
            #data['{}_grids'.format(dataset)][i - offset] = decade_grids
            data['{}_sites'.format(dataset)][i - offset] = decade_site_data

    if save_location is not None:

        print("Saving to: {}".format(save_location))

        np.save(os.path.join(save_location, 'tc_data_train_{}'.format(file_time)), np.array(all_train_tc_data, dtype=object), allow_pickle=True)
        np.save(os.path.join(save_location, 'tc_data_test_{}'.format(file_time)), np.array(all_test_tc_data, dtype=object), allow_pickle=True)

    print("Training Data Generation Complete")
    return all_train_inputs, all_train_outputs, all_test_inputs, all_test_outputs


if __name__ == "__main__":
    generateTrainingData(5, 3, 0, 0, './Data')
    parser = argparse.ArgumentParser(description='Generate machine learning training data from STORM')
    parser.add_argument('total_years',  type=int,
                    help='Number of years to run STORM for when generating training data')
    parser.add_argument('num_training', type=int,
                    help='Number of training samples')
    parser.add_argument('num_test',  type=int,
                    help='number of test samples')
    parser.add_argument('num_validation', type=int,
                        help='number of validation samples')
    parser.add_argument('save_location', type=str,
                    help='Directory to save data to.')
    args = parser.parse_args()

    generateTrainingData(
        args.total_years,
        args.num_training,
        args.num_test,
        args.num_validation,
        args.save_location
    )

