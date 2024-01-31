from SampleSTORM import sampleStorm
from RiskFactors import averageLandfallsPerMonth
from GenerateInputParameters import generateInputParameters
import os
import numpy as np
import argparse

import cProfile
import h5py

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

monthsall=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

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

    refs.append(genesis_matrix)
    refs.append(movement_coefficients)

    tc_data = sampleStorm(total_years, month_map, refs)

    avg_landfalls_per_month = averageLandfallsPerMonth(tc_data, basin, total_years, .5)
    basin_movement_coefficients = movement_coefficients[basins.index(basin)]

    # split up input, output data for each month and flatten the matrices
    genesis_matrix = np.nan_to_num(genesis_matrix)



    return (genesis_matrix, basin_movement_coefficients), avg_landfalls_per_month, tc_data


def generateTrainingData(total_years, n_train_samples, n_test_samples, save_location, basin='SP'):

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

    land_mask=np.loadtxt(os.path.join(__location__, 'STORM', 'Land_ocean_mask_'+str(basin)+'.txt'))

    mu_list=np.loadtxt(os.path.join(__location__,'STORM', 'POISSON_GENESIS_PARAMETERS.txt'))

    monthlist=np.load(os.path.join(__location__,'STORM','GENESIS_MONTHS.npy'), allow_pickle=True).item()

    movementCoefficientsFuture = [np.load(os.path.join(__location__, 'InputData', "JM_LONLATBINS_IBTRACSDELTA_{}.npy".format(model))
                                          , allow_pickle=True)
                                  .item()['SP']
                                  for model in ['CMCC-CM2-VHR4','EC-Earth3P-HR','CNRM-CM6-1-HR','HadGEM3-GC31-HM']]

    models = ['CMCC-CM2-VHR4', 'EC-Earth3P-HR', 'CNRM-CM6-1-HR', 'HadGEM3-GC31-HM']
    future_delta_files = [os.path.join(__location__, 'InputData', "GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model))
                          for model in models]

    future_data = [np.load(file_path, allow_pickle=True).item()[basin] for file_path in future_delta_files]

    data = h5py.File(os.path.join(save_location, 'AllData.hdf5'), 'w')

    lat, lon = future_data[0][1].shape

    genesis_train_data = data.create_dataset('train_genesis', (n_train_samples, 6, lat, lon))
    genesis_test_data = data.create_dataset('test_genesis', (n_test_samples, 6, lat, lon))

    w = len(movementCoefficientsFuture[0])
    h = len(movementCoefficientsFuture[0][0])
    movement_coefficient_train_data = data.create_dataset('train_movement', (n_train_samples, w, h))
    movement_coefficient_test_data = data.create_dataset('test_movement', (n_test_samples, w, h))

    output_train_data = data.create_dataset('train_output', (n_train_samples, 2*lat, 2*lon, 12))
    output_test_data = data.create_dataset('test_output', (n_test_samples, 2*lat, 2*lon, 12))


    rmax_pres = np.load(os.path.join(__location__, 'STORM', 'RMAX_PRESSURE.npy'),allow_pickle=True).item()

    print("Finished loading files")

    all_train_tc_data = []
    all_test_tc_data = []

    print("Generating training samples.")
    for i in range(n_train_samples):
        input, output, tc_data = generateOneTrainingDataSample(
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
            rmax_pres
            ],
            basin
        )
        genesis_matrices, movement_coefficients = input
        genesis_train_data[i] = genesis_matrices
        movement_coefficient_train_data[i] = movement_coefficients
        output_train_data[i] = output

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
                JM_pressure_for_basin,
                Genpres_for_basin,
                WPR_coefficients_for_basin,
                Genwind_for_basin,
                Penv,
                land_mask,
                mu_list,
                monthlist,
                rmax_pres
            ],
            basin
        )

        genesis_matrices, movement_coefficients = input
        genesis_test_data[i] = genesis_matrices
        movement_coefficient_test_data[i] = movement_coefficients
        output_test_data[i] = output

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

        np.save(os.path.join(save_location, 'train', 'tc_data'), np.array(all_train_tc_data, dtype=object), allow_pickle=True)
        np.save(os.path.join(save_location, 'test', 'tc_data'), np.array(all_test_tc_data, dtype=object), allow_pickle=True)

    data.close()
    print("Training Data Generation Complete")
    return all_train_inputs, all_train_outputs, all_test_inputs, all_test_outputs

print(__name__)
if __name__ == "__main__":

    print("CALLED", flush=True)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('total_years',  type=int,
                    help='Number of years to run STORM for when generating training data')
    parser.add_argument('num_training', type=int,
                    help='Number of training samples')
    parser.add_argument('num_test',  type=int,
                    help='number of test samples')
    parser.add_argument('save_location', type=str,
                    help='Directory to save data to.')
    args = parser.parse_args()


    cProfile.run("generateTrainingData(args.total_years, args.num_training, args.num_test, args.save_location)")
