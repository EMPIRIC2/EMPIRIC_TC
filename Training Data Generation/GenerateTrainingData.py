
from SampleSTORM import sampleStorm
from RiskFactors import averageLandfallsPerMonth
from GenerateInputParameters import generateInputParameters

import numpy as np

import tensorflow as tf

monthsall=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

def convert_to_flat_inputs(genesis_matrix, movement_coefficients):

    # flatten and concatenate the inputs
    inputs = np.concatenate((genesis_matrix.flatten(), np.array(movement_coefficients).flatten()))

    return inputs

def convert_to_flat_outputs(avg_landfalls_per_month):
    return avg_landfalls_per_month.flatten()


def generateOneTrainingDataSample(total_years, convert_inputs_to_nn_format, convert_outputs_to_nn_format, basin='SP'):
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

    avg_landfalls_per_month = averageLandfallsPerMonth(tc_data, basin, total_years, 1)

    basin_movement_coefficients = movement_coefficients[basins.index(basin)]

    # split up input, output data for each month and flatten the matrices
    genesis_matrix = np.array([np.nan_to_num(genesis_matrices[month]) for month in monthlist])

    X = convert_inputs_to_nn_format(genesis_matrix, basin_movement_coefficients)

    Y = convert_outputs_to_nn_format(avg_landfalls_per_month)
    return X, Y

def generateTrainingData(total_years, n_train_samples, n_test_samples, convert_inputs_to_nn_format, convert_outputs_to_nn_format, basin='SP', save_location=None):

    all_train_inputs = []
    all_train_outputs = []

    all_test_inputs = []
    all_test_outputs = []

    for i in range(n_train_samples):
        input, output = generateOneTrainingDataSample(
            total_years,
            convert_inputs_to_nn_format,
            convert_outputs_to_nn_format,
            basin
        )

        all_train_inputs.append(input)
        all_train_outputs.append(output)

    for i in range(n_test_samples):
        input, output, = generateOneTrainingDataSample(
            total_years,
            convert_inputs_to_nn_format,
            convert_outputs_to_nn_format,
            basin
        )
        all_test_inputs.append(input)
        all_test_outputs.append(output)

    if save_location is not None:
        train = (np.array(all_train_inputs), np.array(all_train_outputs))

        test = (np.array(all_test_inputs), np.array(all_test_outputs))

        train_dataset = tf.data.Dataset.from_tensor_slices(train)
        test_dataset = tf.data.Dataset.from_tensor_slices(test)
        train_dataset.save(save_location)
        test_dataset.save(save_location)

    return all_train_inputs, all_train_outputs, all_test_inputs, all_test_outputs


generateTrainingData(1, 10, 5, convert_to_flat_inputs, convert_to_flat_outputs, save_location='./Data/')
