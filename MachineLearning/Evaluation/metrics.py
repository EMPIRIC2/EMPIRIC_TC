from utils import _get_inputs, _get_outputs
from model_statistics import compute_ensemble_statistics
from figures import make_figures
from figures import save_metrics_as_latex

from MachineLearning.dataset import get_dataset
from MachineLearning.UNet.unet import UNet


from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from relative_change_metrics import compute_all_relative_change_pairs
from model_statistics import KS_statistics
from site_metrics import total_site_mse

def compute_metrics(ground_outputs, model_outputs, ground_statistics, model_statistics, model_name):
    """
    Computes all the metrics for a given model and statistics

    :param ground_outputs: the "true" model outputs to benchmark against (i.e. STORM->RISK model)
    :param model_outputs: the outputs of the model to benchmark (i.e. the ML model)
    :param ground_statistics: the statistics computed from the output of the ground model
    :param model_statistics: the statistics computed from the output of model being evaluated
    :param model_name: the name of the model that we are evaluating. Meant for displaying table of multiple model performance

    :returns: a dict containing model name and all the metrics used to evaluate the model
    """

    top_relative_change_error_maps, mse = compute_all_relative_change_pairs(ground_outputs, model_outputs)

    # Compute metrics from the statistics
    metrics = {
        "Model": model_name,
        "Mean Absolute Quantile Error":  mean_absolute_error(ground_statistics["Quantiles"].flatten(), model_statistics["Quantiles"].flatten()),
        "Mean Squared Quantile Error":  mean_squared_error(ground_statistics["Quantiles"].flatten(), model_statistics["Quantiles"].flatten()),
        "Kolmogorov-Smirnov": KS_statistics(ground_outputs, model_outputs),
        "Relative Change Mean Squared Error": mse,
        "Relative Error Examples": top_relative_change_error_maps,
        "Site Mean Squared Error": total_site_mse(ground_outputs, model_outputs)
    }

    return metrics

def compute_test_statistics(data_folder, model_path, output_save_folder):
    """
    Master function to compute all the statistics for a particular model and then save the figures

    :param: data_folder: path to the data used to evaluate the model
    :param: model_path: path to the model weights to evaluate
    :param: output_save_folder: folder to save metrics latex and figure pictures

    :returns: None. But saves files.
    """

    genesis_size_default=(55, 105, 1)
    movement_size_default = (11, 13)

    test_data = get_dataset(data_folder, data_version=3, dataset='test', batch_size=32)

    model = UNet(genesis_size_default, movement_size_default, 1)

    model.load_weights(model_path)

    outputs = _get_outputs(test_data)
    inputs = _get_inputs(test_data)

    predictions = model.predict(
        inputs,
        batch_size=32,
        verbose=2,
        steps=1
    )

    storm_statistics = compute_ensemble_statistics(outputs)
    unet_statistics = compute_ensemble_statistics(predictions)

    metrics = compute_metrics(outputs, predictions, storm_statistics, unet_statistics, "UNet Custom")

    save_metrics_as_latex([metrics], os.path.join(output_save_folder, "metrics.tex"))

    make_figures(outputs, predictions, storm_statistics, unet_statistics, metrics, output_save_folder)
