
from SampleSTORM import sampleStorm
from RiskFactors import averageLandfallsPerMonth
from GenerateInputParameters import generateInputParameters
from PlotMapData import plotLatLonGridData
import ray
import os
import numpy as np

import tensorflow as tf

monthsall=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

@ray.remote
def generateOneTrainingDataSample(total_years,  basin='SP'):
    '''
    Generate ML training data

    The ML inputs, X, are randomly generated inputs to the STORM model
    The output targets, Y, are grids of average landfall per month

    :param total_years: number of years to run the STORM simulations over
    :return:
    '''

    basins = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']

    monthlist = monthsall[basins.index(basin)]

    genesis_matrices, movement_coefficients = generateInputParameters(basin, monthlist) # replace with generated parameters

    tc_data = sampleStorm(total_years, genesis_matrices, movement_coefficients)

    avg_landfalls_per_month = averageLandfallsPerMonth(tc_data, basin, total_years, .5)

    basin_movement_coefficients = movement_coefficients[basins.index(basin)]

    # split up input, output data for each month and flatten the matrices
    genesis_matrix = np.array([np.nan_to_num(genesis_matrices[month]) for month in monthlist])

    plotLatLonGridData(genesis_matrices[1], 1)

    plotLatLonGridData(np.flipud(np.sum(avg_landfalls_per_month, axis=2)), .5)

    return (genesis_matrix, basin_movement_coefficients), avg_landfalls_per_month


def generateTrainingData(total_years, n_train_samples, n_test_samples, basin='SP', save_location=None):
    all_train_inputs = []
    all_train_outputs = []

    all_test_inputs = []
    all_test_outputs = []

    training_sample_refs = []
    MAX_NUM_PENDING_TASKS = 20

    for i in range(n_train_samples):

        if len(training_sample_refs) > MAX_NUM_PENDING_TASKS:

            # update result_refs to only
            # track the remaining tasks.
            ready_refs, training_sample_refs = ray.wait(training_sample_refs, num_returns=1)

            input, output = ray.get(ready_refs[0])

            all_train_inputs.append(input)
            all_train_outputs.append(output)

        training_sample_refs.append(generateOneTrainingDataSample.remote(
            total_years,
            basin
        ))

    test_sample_refs = []
    for i in range(n_test_samples):

        if len(test_sample_refs) > MAX_NUM_PENDING_TASKS:
            ready_refs, test_sample_refs = ray.wait(test_sample_refs, num_returns=1)

            input, output = ray.get(ready_refs[0])


            all_test_inputs.append(input)
            all_test_outputs.append(output)

        test_sample_refs.append(generateOneTrainingDataSample.remote(
            total_years,
            basin
        ))

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

    if save_location is not None:

        genesis_matrices, movement_coefficients = zip(*all_train_inputs)

        train = (np.array(genesis_matrices), np.array(movement_coefficients), np.array(all_train_outputs))

        test_genesis_matrices, test_movement_coefficients = zip(*all_test_inputs)
        test = (np.array(test_genesis_matrices), np.array(test_movement_coefficients), np.array(all_test_outputs))

        train_dataset = tf.data.Dataset.from_tensor_slices(train)
        test_dataset = tf.data.Dataset.from_tensor_slices(test)

        train_dataset.save(os.path.join(save_location, 'train'))
        test_dataset.save(os.path.join(save_location, 'test'))

    return all_train_inputs, all_train_outputs, all_test_inputs, all_test_outputs

generateTrainingData(1, 1, 1, save_location='./Data/')