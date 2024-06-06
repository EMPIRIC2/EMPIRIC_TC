from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import numpy as np

def relative_change(a, b, eps=1e-5):
    """
    Returns the relative change from a to b.
    """

    return (a - b) / (a + eps)

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

        if i > max_pairs:
            break
        j, k = pair
        ground_output_1 = ground_outputs[j]
        ground_output_2 = ground_outputs[k]

        ground_change = relative_change(ground_output_1, ground_output_2)

        model_output_1 = model_outputs[j]
        model_output_2 = model_outputs[k]

        model_change = relative_change(model_output_1, model_output_2)

        error_map = ground_change - model_change
        mse = mean_squared_error(ground_change, model_change)

        error_maps.append(error_map)
        mean_squared_errors.append(mse)

    total_mse = np.mean(mean_squared_errors)

    n_examples = min(10, len(ground_outputs))
    largest_error_indices = np.argpartition(mean_squared_errors, -n_examples)[-n_examples:]
    top_error_maps = np.array(error_maps)[largest_error_indices]

    return top_error_maps, total_mse
