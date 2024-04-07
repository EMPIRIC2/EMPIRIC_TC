
from MachineLearning.ConvProbPredictor.conv_prob_predictor import *
from MachineLearning.dataset import get_dataset
import numpy as np
import os

def make_site_predictions(data_folder, weight_path, prediction_save_folder, index=0):
    print("getting data")

    train_data = get_dataset(data_folder, "MachineLearning/ConvProbPredictor/", data_version=2, dataset="test").take(1000)

    print("loading model")

    genesis_shape = (4,)
    movement_shape = (4,)
    num_outputs = 2
    initial_biases = np.load('MachineLearning/ConvProbPredictor/initial_biases_new.npy')

    model = conv_prob_predictor(genesis_shape, movement_shape, num_outputs, initial_biases)

    model.load_weights(weight_path)

    samples = [item for i, item in enumerate(train_data.as_numpy_iterator()) if i < 10]

    # group outputs by the input genesis matrix
    outputs = [samples[i][1][j] for i in range(len(samples)) for j in range(len(samples[i][1]))]

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

#make_site_predictions("/nesi/project/uoa03669/ewin313/storm_data/v2/", "/nesi/project/uoa03669/ewin313/TropicalCycloneAI/models/site_prob_1708895505.3277183.weights.h5", "./predictions/")
