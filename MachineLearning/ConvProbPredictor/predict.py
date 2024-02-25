
from conv_prob_predictor import *
from MachineLearning.dataset import get_dataset
import numpy as np
import os

def make_site_predictions(data_folder, weight_path, prediction_save_folder, index=0):
    print("getting data")

    train_data = get_dataset(data_folder, data_version=2)

    print("loading model")

    genesis_shape = (55, 105, 1)
    movement_shape = (13,)
    num_outputs = 542

    model = conv_prob_predictor(genesis_shape, movement_shape, num_outputs)
    model.load_weights(weight_path)

    samples = [item for i, item in enumerate(train_data.as_numpy_iterator()) if i < 100]

    # group outputs by the input genesis matrix
    outputs = []
    outputs_for_genesis = []
    for i in range(len(samples)):
        if i != 0 and samples[i][0] != samples[i-1][0]:
            outputs.append(outputs_for_genesis)
            outputs_for_genesis = []
        outputs_for_genesis.append(samples[i][1])

        if i == len(samples) - 1:
            outputs.append(outputs_for_genesis)


    print(outputs)
    print("making predictions")

    predictions = model.predict(
        train_data,
        batch_size=1,
        verbose=2,
        steps=1
    )

    np.save(os.path.join(prediction_save_folder, "model_predictions{}.npy".format(index)), predictions)
    np.save(os.path.join(prediction_save_folder, "real_outputs{}.npy".format(index)), outputs)

    return predictions, outputs

make_site_predictions("../../Training Data Generation/Data/v3", "../models/site_prob_1708640562.211484.weights.h5", "./predictions/")