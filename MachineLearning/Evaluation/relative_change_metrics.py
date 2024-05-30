from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import numpy as np

def relative_change(a, b):
    """
    Returns the relative change from a to b.
    """

    return (a - b) / (a + 1e-3)

def compute_changes_between_2_samples(ground_outputs, model_outputs, i, j):
    """
    given two inputs and two models, compute the error in relative (percentage) change between two inputs, between the models
    i.e. this provides a measure of how well the model captures changes between two different inputs compared to the ground model (STORM)

    :param ground_outputs: the outputs from the ground truth model (STORM -> RISK)
    :param model_outputs: the outputs from the model being evaluated
    :param i: index of first example
    :param j: index of second example
    """

    ground_output_1 = ground_outputs[i]
    ground_output_2 = ground_outputs[j]

    ground_change = relative_change(ground_output_1, ground_output_2)

    model_output_1 = model_outputs[i]
    model_output_2 = model_outputs[j]

    model_change = relative_change(model_output_1, model_output_2)

    error_map = ground_change - model_change
    print(ground_change.shape)
    return error_map, mean_squared_error(ground_change, model_change)

def compute_all_relative_change_pairs(ground_outputs, model_outputs, max_pairs = 200):
    """
    Compute the error in relative output changes for all pairs outputs

    return: maps of the relative errors with the largest 10 mean squared errors and the total mean squared error (over all pair results)
    """
    
    assert len(ground_outputs) == len(model_outputs)
    
    pairs = itertools.combinations(range(len(ground_outputs)), 2)

    error_maps = []
    mean_squared_errors = []
    for i, pair in enumerate(pairs):
        if i > max_pairs: break
        error_map, mse = compute_changes_between_2_samples(ground_outputs, model_outputs, *pair)
        error_maps.append(error_map)
        mean_squared_errors.append(mse)

    total_mse = np.mean(mean_squared_errors)

    n_examples = min(10, len(ground_outputs))
    largest_error_indices = np.argpartition(mean_squared_errors, -n_examples)[-n_examples:]
    top_error_maps = np.array(error_maps)[largest_error_indices]

    return top_error_maps, total_mse
