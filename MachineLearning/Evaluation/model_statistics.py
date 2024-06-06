import numpy as np
import scipy

def get_quantiles(data):
    return np.quantile(data, [0, 0.25,.5,.75,1], axis=0)

def compute_ensemble_statistics(outputs):
    statistics = {
        "Quantiles": get_quantiles(outputs),
        "Mean": np.mean(outputs, axis=0)
    }

    return statistics

def kolmogorov_smirnov_statistics(ground_truths, predictions):
    """
    Compute the kolmogorov smirnov statistic across the ensemble of data

    :param: ground_truths: the outputs from the model being used as the baseline (i.e. STORM) (a list of n x m numpy arrays)
    :param: predictions: outputs of the model being evaluated (a list of n x m numpy arrays of the same length as ground_truths)

    :returns: a grid of ks statistics with shape n x m. i.e. ks_statistic[i, j]
    is the ks statistic comparing the distribution of values at ground_truths[i, j]
    with the distribution of values at predictions[i, j]
    """
    ks_statistics = []

    # flatten all but the first axis
    sample_shape = ground_truths[0].shape
    ground_truth_sample_length = sample_shape[0] * sample_shape[1]
    ground_truths = np.reshape(ground_truths, (len(ground_truths), ground_truth_sample_length))
    predictions = np.reshape(predictions, (len(ground_truths), ground_truth_sample_length))

    # compute the kolmogorov smirnov statistic
    for i in range(len(ground_truths[0])):
        ks_statistics.append(scipy.stats.kstest(ground_truths[:,i], predictions[:,i]).statistic)

    # regrid the ks statistics
    ks_statistics = np.reshape(ks_statistics, sample_shape)
    return ks_statistics