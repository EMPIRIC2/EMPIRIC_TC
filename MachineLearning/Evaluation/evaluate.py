import argparse
from MachineLearning.Evaluation.evaluation_utils import process_predictions
from MachineLearning.Evaluation.model_statistics import compute_ensemble_statistics
from MachineLearning.Evaluation.metrics import compute_metrics
from MachineLearning.Evaluation.model_info import models_info
from MachineLearning.Evaluation.figures import make_figures, save_metrics_as_latex
from MachineLearning.dataset import get_dataset
import numpy as np
import os

def evaluate(dataset, output_dir, predict_params):
    """
    Compute all evaluation metrics and then save the resulting figures to disk.

    :param: data_folder: path to the data used to evaluate the model
    :param: model_path: path to the model weights to evaluate
    :param: output_save_folder: folder to save metrics latex and figure pictures
    """

    outputs_ds = dataset.map(lambda x, y: y)
    outputs = np.squeeze(np.concatenate(list(outputs_ds.as_numpy_iterator()), axis=0))

    inputs = dataset.map(lambda x,y: x)

    metrics = []
    storm_statistics = compute_ensemble_statistics("STORM", outputs)

    for model_info in models_info:
        model = model_info["model"]

        predictions = model.predict(
            inputs,
            **predict_params
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

    test_data = get_dataset(args.data_folder, data_version=4, dataset='test', batch_size=32)

    predict_params = {
        "batch_size": 32,
        "verbose": 2
    }

    evaluate(test_data, args.output_save_folder, predict_params)
