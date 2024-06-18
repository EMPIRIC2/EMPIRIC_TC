from sklearn.metrics import mean_absolute_error, mean_squared_error

from MachineLearning.Evaluation.model_statistics import \
    kolmogorov_smirnov_statistics
from MachineLearning.Evaluation.relative_change_metrics import \
    compute_all_relative_change_pairs
from MachineLearning.Evaluation.site_metrics import \
    total_site_mean_squared_error


def compute_metrics(
    ground_outputs, model_outputs, ground_statistics, model_statistics, model_name
):
    """
    Computes all the metrics for a given model and statistics

    :param ground_outputs: the "true" model outputs to benchmark
    against (i.e. STORM->RISK model)
    :param model_outputs: the outputs of the model to
    benchmark (i.e. the ML model)
    :param ground_statistics: the statistics computed from
    the output of the ground model
    :param model_statistics: the statistics computed from the output
    of model being evaluated
    :param model_name: the name of the model that we are evaluating.
    Meant for displaying table of multiple model performance

    :returns: a dict containing model name and all the metrics
    used to evaluate the model
    """

    top_relative_change_error_maps, mse = compute_all_relative_change_pairs(
        ground_outputs, model_outputs
    )

    # Compute metrics from the statistics
    metrics = {
        "Model": model_name,
        "Mean Absolute Quantile Error": mean_absolute_error(
            ground_statistics["Quantiles"].flatten(),
            model_statistics["Quantiles"].flatten(),
        ),
        "Mean Squared Quantile Error": mean_squared_error(
            ground_statistics["Quantiles"].flatten(),
            model_statistics["Quantiles"].flatten(),
        ),
        "Kolmogorov-Smirnov": kolmogorov_smirnov_statistics(
            ground_outputs, model_outputs
        ),
        "Relative Change Mean Squared Error": mse,
        "Relative Error Examples": top_relative_change_error_maps,
        "Site Mean Squared Error": total_site_mean_squared_error(
            ground_outputs, model_outputs
        ),
    }

    return metrics
