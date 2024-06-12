import argparse
from MachineLearning.Evaluation.evaluation_utils import process_predictions
from MachineLearning.Evaluation.model_statistics import compute_ensemble_statistics
from MachineLearning.Evaluation.metrics import compute_metrics
from MachineLearning.Evaluation.figures import make_figures, save_metrics_as_latex
from MachineLearning.dataset import get_dataset
from MachineLearning.NearestNeighbors.nearest_neighbors import NearestNeighborsRegressor
import numpy as np
import os
from saved_models.saved_models import DDPMUNet02CatCyclones, DDPMUNetNoAttention02CatCyclones, UNetCustom02CatCyclones, SongUNet

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def evaluate(outputs_ds, output_dir):
    """
    Compute all evaluation metrics and then save the resulting figures to disk.

    :param: data_folder: path to the data used to evaluate the model
    :param: model_path: path to the model weights to evaluate
    :param: output_save_folder: folder to save metrics latex and figure pictures
    """

    metrics = []

    batch_axis = 0
    unbatched_outputs = np.concatenate(list(outputs_ds.as_numpy_iterator()), axis=batch_axis)
    outputs = np.squeeze(unbatched_outputs)
    storm_statistics = compute_ensemble_statistics("STORM", outputs)

    for model_info in models_info:

        model = model_info["model"]
        inputs = model_info["inputs"]

        predictions = model.predict(
            inputs
        )
        
        predictions = process_predictions(predictions)

        model_statistics = compute_ensemble_statistics(model_info["Name"], predictions)

        model_metrics = compute_metrics(outputs, predictions, storm_statistics, model_statistics, model_info["Name"])
        metrics.append(model_metrics)

        if not os.path.exists(os.path.join(output_dir, model_info["Name"])):
            os.makedirs(os.path.join(output_dir, model_info["Name"]))

        make_figures(outputs, predictions, storm_statistics, model_statistics, model_metrics,
                     os.path.join(output_dir, model_info["Name"]))

    save_metrics_as_latex(metrics, os.path.join(output_dir, "metrics.tex"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Evaluate Models",
        description="Evaluates trained models in models_info against STORM"
    )

    parser.add_argument("data_folder", type=str)
    parser.add_argument("output_save_folder", type=str)

    args = parser.parse_args()

    test_data_ml = get_dataset(args.data_folder, data_version=4, dataset='test', batch_size=32)

    outputs_ds = test_data_ml.map(lambda x, y: y)
    ml_inputs = test_data_ml.map(lambda x, y: x)

    test_data_nearest_neighbors = get_dataset(args.data_folder, data_version=5, dataset='test', batch_size=32)
    nearest_neighbors_inputs = test_data_nearest_neighbors.map(lambda x, y: x)
    nearest_neighbors_regressor = NearestNeighborsRegressor(args.data_folder)
    nearest_neighbors_regressor.load(os.path.join(__location__, '../NearestNeighbors/nearest_neighbors.pkl'))
    """
    Object keeps track of the models we want to evaluate.
    """
    models_info = [
        {
            "Name": "DDIM Unet",
            "Output": "Mean 0-2 Category TCs over 10 years",
            "model": SongUNet.load_model(),
            "inputs": ml_inputs
        },
        {
            "Name": "Custom UNet",
            "Output": "Mean 0-2 Category TCs over 10 years",
            "model": UNetCustom02CatCyclones.load_model(),
            "inputs": ml_inputs
        },
        {
            "Name": "DDPM UNet",
            "Output": "Mean 0-2 Category TCs over 10 years",
            "model": DDPMUNet02CatCyclones.load_model(),
            "inputs": ml_inputs
        },
        {
            "Name": "DDPM UNet w/o attention",
            "Output": "Mean 0-2 Category TCs over 10 years",
            "model": DDPMUNetNoAttention02CatCyclones.load_model(),
            "inputs": ml_inputs
        },
        {
            "Name": "Nearest Neighbors Regressor",
            "Output": "Mean 0-2 Category TCs over 10 years",
            "model": nearest_neighbors_regressor,
            "inputs": nearest_neighbors_inputs
        }
    ]

    predict_params = {
        "batch_size": 32,
        "verbose": 2
    }

    evaluate(outputs_ds, args.output_save_folder)
