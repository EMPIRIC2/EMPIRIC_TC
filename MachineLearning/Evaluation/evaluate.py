import argparse
import os

import numpy as np

from MachineLearning.dataset import get_dataset
from MachineLearning.Evaluation.evaluation_utils import process_predictions
from MachineLearning.Evaluation.figures import (make_collective_model_figures,
                                                make_single_model_figures,
                                                save_metrics_as_latex)
from MachineLearning.Evaluation.metrics import compute_metrics
from MachineLearning.Evaluation.model_statistics import \
    compute_ensemble_statistics
from MachineLearning.NearestNeighbors.nearest_neighbors import \
    NearestNeighborsRegressor
from saved_models.saved_models import (DDPMUNet02CatCyclones,
                                       DDPMUNetNoAttention02CatCyclones,
                                       SongUNet, UNetCustom02CatCyclones)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def evaluate(outputs_ds, output_dir, models):
    """
    Compute all evaluation metrics and then save the resulting figures to disk.

    :param: data_folder: path to the data used to evaluate the model
    :param: model_path: path to the model weights to evaluate
    :param: output_save_folder: folder to save metrics latex and figure pictures
    """

    metrics = []

    batch_axis = 0

    unbatched_outputs = np.concatenate(
        list(outputs_ds.as_numpy_iterator()), axis=batch_axis
    )
    outputs = np.squeeze(unbatched_outputs)
    storm_statistics = compute_ensemble_statistics("STORM", outputs)

    all_outputs = {"STORM": outputs}
    all_statistics = [storm_statistics]

    for model_info in models:
        model = model_info["model"]
        inputs = model_info["inputs"]

        predictions = model.predict(inputs)

        predictions = process_predictions(predictions)

        model_statistics = compute_ensemble_statistics(model_info["name"], predictions)

        model_metrics = compute_metrics(
            outputs, predictions, storm_statistics, model_statistics, model_info["name"]
        )
        metrics.append(model_metrics)

        if not os.path.exists(os.path.join(output_dir, model_info["name"])):
            os.makedirs(os.path.join(output_dir, model_info["name"]))

        make_single_model_figures(
            outputs,
            predictions,
            storm_statistics,
            model_statistics,
            model_metrics,
            os.path.join(output_dir, model_info["name"]),
        )

        all_outputs[model_info["name"]] = predictions
        all_statistics.append(model_statistics)

    make_collective_model_figures(all_outputs, all_statistics, output_dir)
    save_metrics_as_latex(metrics, os.path.join(output_dir, "metrics.tex"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Evaluate Models",
        description="Evaluates trained models in models_info against STORM",
    )

    parser.add_argument("eval_data_dir", type=str)
    parser.add_argument("train_data_dir", type=str)
    parser.add_argument("output_save_dir", type=str)

    args = parser.parse_args()

    test_data_ml = get_dataset(
        args.eval_data_dir, data_version=4, dataset="test", batch_size=32
    )

    outputs_ds = test_data_ml.map(lambda x, y: y)
    ml_inputs = test_data_ml.map(lambda x, y: x)

    test_data_nearest_neighbors = get_dataset(
        args.eval_data_dir, data_version=5, dataset="test", batch_size=32
    )
    nearest_neighbors_inputs_ds = test_data_nearest_neighbors.map(lambda x, y: x)

    nearest_neighbors_inputs = np.empty(shape=(0, 5775))
    for data in nearest_neighbors_inputs_ds.as_numpy_iterator():
        nearest_neighbors_inputs = np.concatenate([nearest_neighbors_inputs, data])

    nearest_neighbors_regressor = NearestNeighborsRegressor(args.train_data_dir)

    nearest_neighbors_regressor.load(
        os.path.join(__location__, "../NearestNeighbors/nearest_neighbors.pkl")
    )
    """
    Object keeps track of the models we want to evaluate.
    """
    models_info = [
        {
            "name": "Nearest Neighbors Regressor",
            "output_description": "Mean 0-2 Category TCs over 10 years",
            "model": nearest_neighbors_regressor,
            "inputs": nearest_neighbors_inputs,
        },
        {
            "name": "DDIM Unet",
            "output_description": "Mean 0-2 Category TCs over 10 years",
            "model": SongUNet.load_model(),
            "inputs": ml_inputs,
        },
        {
            "name": "Custom UNet",
            "output_description": "Mean 0-2 Category TCs over 10 years",
            "model": UNetCustom02CatCyclones.load_model(),
            "inputs": ml_inputs,
        },
        {
            "name": "DDPM UNet",
            "output_description": "Mean 0-2 Category TCs over 10 years",
            "model": DDPMUNet02CatCyclones.load_model(),
            "inputs": ml_inputs,
        },
        {
            "name": "DDPM UNet w/o attention",
            "output_description": "Mean 0-2 Category TCs over 10 years",
            "model": DDPMUNetNoAttention02CatCyclones.load_model(),
            "inputs": ml_inputs,
        },
    ]

    predict_params = {"batch_size": 32, "verbose": 2}

    evaluate(outputs_ds, args.output_save_folder, models_info)
