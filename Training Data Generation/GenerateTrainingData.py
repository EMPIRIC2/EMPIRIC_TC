
from SampleSTORM import sampleStorm
from RiskFactors import averageLandfallsPerMonth
from GenerateInputParameters import generateInputParameters
from PlotMapData import plotLatLonGridData
import cProfile
import ray
import os
import numpy as np
import time
import argparse


import tensorflow as tf

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

monthsall=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

#@ray.remote
def generateOneTrainingDataSample(total_years, future_data, movementCoefficientsFuture, refs, basin='SP'):
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
    genesis_matrix_ref = ray.put(genesis_matrix)

    movement_coefficients_ref = ray.put(movement_coefficients)

    refs.append(genesis_matrix_ref)
    refs.append(movement_coefficients_ref)

    tc_data = sampleStorm(total_years, month_map, refs)

    avg_landfalls_per_month = averageLandfallsPerMonth(tc_data, basin, total_years, .5)

    basin_movement_coefficients = movement_coefficients[basins.index(basin)]

    # split up input, output data for each month and flatten the matrices
    genesis_matrix = np.nan_to_num(genesis_matrix)


    return (genesis_matrix, basin_movement_coefficients), avg_landfalls_per_month, tc_data


def generateTrainingData(total_years, n_train_samples, n_test_samples, basin='SP', save_location=None):

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

    JM_pressure_ref = ray.put(JM_pressure_for_basin)

    Genpres = np.load(os.path.join(__location__, 'STORM', 'DP0_PRES_GENESIS.npy'), allow_pickle=True).item()
    Genpres_for_basin = np.array([Genpres[basins.index(basin)][month] for month in monthlist])

    Genpres_ref = ray.put(Genpres_for_basin)
    # this is the wind pressure relationship coefficients: eq. 3 in the
    WPR_coefficients = np.load(os.path.join(__location__, 'STORM', 'COEFFICIENTS_WPR_PER_MONTH.npy'),
                               allow_pickle=True).item()
    WPR_coefficients_for_basin = np.array([WPR_coefficients[basins.index(basin)][month] for month in monthlist])

    WPR_coefficients_ref = ray.put(WPR_coefficients_for_basin)

    Genwind = np.load(os.path.join(__location__, 'STORM', 'GENESIS_WIND.npy'), allow_pickle=True).item()
    Genwind_for_basin = [Genwind[basins.index(basin)][month] for month in monthlist]
    Genwind_ref = ray.put(Genwind_for_basin)

    Penv = {month: np.loadtxt(os.path.join(__location__, 'STORM', 'Monthly_mean_MSLP_' + str(month) + '.txt')) for month in monthlist}
    Penv_ref = ray.put(Penv)

    land_mask=np.loadtxt(os.path.join(__location__, 'STORM', 'Land_ocean_mask_'+str(basin)+'.txt'))
    land_mask_ref = ray.put(land_mask)

    mu_list=np.loadtxt(os.path.join(__location__,'STORM', 'POISSON_GENESIS_PARAMETERS.txt'))
    mu_list_ref = ray.put(mu_list)

    monthlist=np.load(os.path.join(__location__,'STORM','GENESIS_MONTHS.npy'), allow_pickle=True).item()
    monthlist_ref = ray.put(monthlist)

    movementCoefficientsFuture = [np.load(os.path.join(__location__, 'InputData', "JM_LONLATBINS_IBTRACSDELTA_{}.npy".format(model))
                                          , allow_pickle=True)
                                  .item()['SP']
                                  for model in ['CMCC-CM2-VHR4','EC-Earth3P-HR','CNRM-CM6-1-HR','HadGEM3-GC31-HM']]

    models = ['CMCC-CM2-VHR4', 'EC-Earth3P-HR', 'CNRM-CM6-1-HR', 'HadGEM3-GC31-HM']
    future_delta_files = [os.path.join(__location__, 'InputData', "GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model))
                          for model in models]

    future_data = [np.load(file_path, allow_pickle=True).item()[basin] for file_path in future_delta_files]

    rmax_pres=np.load(os.path.join(__location__,'STORM','RMAX_PRESSURE.npy'),allow_pickle=True).item()
    rmax_pres_ref = ray.put(rmax_pres)

    print("Finished loading files")

    all_train_tc_data = []
    all_test_tc_data = []

    print("Generating training samples.")
    for i in range(n_train_samples):
        input, output, tc_data = generateOneTrainingDataSample(
            total_years,
            future_data,
            movementCoefficientsFuture,
            [JM_pressure_ref,
            Genpres_ref,
            WPR_coefficients_ref,
            Genwind_ref,
            Penv_ref,
            land_mask_ref,
            mu_list_ref,
            monthlist_ref,
            rmax_pres_ref
            ],
            basin
        )

        all_train_inputs.append(input)
        all_train_outputs.append(output)
        all_train_tc_data.append(tc_data)
        '''
        if len(training_sample_refs) > MAX_NUM_PENDING_TASKS:

            # update result_refs to only
            # track the remaining tasks.
            ready_refs, training_sample_refs = ray.wait(training_sample_refs, num_returns=1)

            input, output = ray.get(ready_refs[0])

            all_train_inputs.append(input)
            all_train_outputs.append(output)

        training_sample_refs.append(generateOneTrainingDataSample.remote(
            total_years,
            [JM_pressure_ref,
            Genpres_ref,
            WPR_coefficients_ref,
            Genwind_ref],
            basin
        ))'''

    test_sample_refs = []
    for i in range(n_test_samples):
        input, output, tc_data = generateOneTrainingDataSample(
            total_years,
            future_data,
            movementCoefficientsFuture,
            [
                JM_pressure_ref,
                Genpres_ref,
                WPR_coefficients_ref,
                Genwind_ref,
                Penv_ref,
                land_mask_ref,
                mu_list_ref,
                monthlist_ref,
                rmax_pres_ref
            ],
            basin
        )

        all_test_inputs.append(input)
        all_test_outputs.append(output)
        all_test_tc_data.append(tc_data)
        '''
        if len(test_sample_refs) > MAX_NUM_PENDING_TASKS:
            ready_refs, test_sample_refs = ray.wait(test_sample_refs, num_returns=1)

            input, output = ray.get(ready_refs[0])


            all_test_inputs.append(input)
            all_test_outputs.append(output)

        test_sample_refs.append(generateOneTrainingDataSample.remote(
            total_years,
            [JM_pressure_ref,
            Genpres_ref,
            WPR_coefficients_ref,
            Genwind_ref],
            basin
        ))
        '''

    '''
    unfinished = training_sample_refs
    while unfinished:
        ready_refs, unfinished = ray.wait(unfinished, timeout=None)

        input, output = ray.get(ready_refs[0])

        all_train_inputs.append(input)
        all_train_outputs.append(output)

    unfinished = test_sample_refs

    while unfinished:

        ready_refs, unfinished = ray.wait(unfinished, timeout=None)

        input, output = ray.get(ready_refs[0])
        all_test_inputs.append(input)
        all_test_outputs.append(output)
    '''

    if save_location is not None:

        print("Saving to: {}".format(save_location))

        genesis_matrices, movement_coefficients = zip(*all_train_inputs)

        train = (np.array(genesis_matrices), np.array(movement_coefficients), np.array(all_train_outputs))

        test_genesis_matrices, test_movement_coefficients = zip(*all_test_inputs)
        test = (np.array(test_genesis_matrices), np.array(test_movement_coefficients), np.array(all_test_outputs))

        train_dataset = tf.data.Dataset.from_tensor_slices(train)
        test_dataset = tf.data.Dataset.from_tensor_slices(test)

        np.save(os.path.join(save_location, 'train', 'tc_data'), np.array(all_train_tc_data, dtype=object), allow_pickle=True)
        np.save(os.path.join(save_location, 'test', 'tc_data'), np.array(all_test_tc_data, dtype=object), allow_pickle=True)

        train_dataset.save(os.path.join(save_location, 'train'))
        test_dataset.save(os.path.join(save_location, 'test'))

    print("Training Data Generation Complete")
    return all_train_inputs, all_train_outputs, all_test_inputs, all_test_outputs

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('total_years', metavar='N', type=int, nargs='1',
                    help='Number of years to run STORM for when generating training data')
parser.add_argument('num_training', metavar='N', type=int, nargs='1',
                    help='Number of training samples')
parser.add_argument('num_test', metavar='N', type=int, nargs='1',
                    help='number of test samples')
parser.add_argument('save_location', metavar='N', type=str, default=os.path.join(__location__, 'Data'),
                    help='Directory to save data to.')
args = parser.parse_args()

generateTrainingData(args.total_years, args.num_training, args.num_test, args.save_location)