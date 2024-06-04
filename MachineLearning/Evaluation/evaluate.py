from MachineLearning.Evaluation.evaluation_utils import get_outputs, process_predictions
from MachineLearning.Evaluation.model_statistics import compute_ensemble_statistics
import os
from MachineLearning.dataset import get_dataset
from MachineLearning.Evaluation.metrics import compute_metrics
import argparse
from MachineLearning.Evaluation.model_info import models_info

def evaluate(data_dir, output_dir):
    """
    Compute all evaluation metrics and then save the resulting figures to disk.

    :param: data_folder: path to the data used to evaluate the model
    :param: model_path: path to the model weights to evaluate
    :param: output_save_folder: folder to save metrics latex and figure pictures
    """

    test_data = get_dataset(data_dir, data_version=3, dataset='test', batch_size=32)

    outputs = get_outputs(test_data)
    inputs = test_data.map(lambda x,y: x)

    metrics = []
    storm_statistics = compute_ensemble_statistics(outputs)

    for model_info in models_info:
        model = model_info["model"]

        predictions = model.predict(
            inputs,
            batch_size=32,
            verbose=2,
        )
        
        predictions = process_predictions(predictions)

        model_statistics = compute_ensemble_statistics(predictions)

        model_metrics = compute_metrics(outputs, predictions, storm_statistics, model_statistics, model_info["Name"])
        metrics.append(model_metrics)

        if not os.path.exists(os.path.join(output_dir, model_info["Name"])):
            os.makedirs(os.path.join(output_dir, model_info["Name"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Evaluate Models",
        description="Evaluates trained models in models_info against STORM"
    )

    parser.add_argument("data_folder", type=str)
    parser.add_argument("output_save_folder", type=str)

    args = parser.parse_args()

    evaluate(args.data_folder, args.output_save_folder)
