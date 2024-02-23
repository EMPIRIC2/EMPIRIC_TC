import tensorflow as tf
from tensorflow import keras
from dataset import get_dataset
import numpy as np
import os

genesis_size = (55, 105, 6)
movement_size = (11, 13)
output_size = (110, 210, 6)

def predictions(model_path, data_folder, prediction_save_folder):
    train_data = get_dataset(data_folder, genesis_size=genesis_size, output_size=output_size)

    model = tf.keras.models.load_model(model_path)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")
    )

    samples = [item for i, item in enumerate(train_data.as_numpy_iterator()) if i < 10]

    outputs = samples[0][1]

    predictions = model.predict(
            train_data,
            batch_size=10,
            verbose=2,
            steps=1
    )

    np.save(os.path.join(prediction_save_folder, "model_predictions3.npy"), predictions)
    np.save(os.path.join(prediction_save_folder, "real_outputs3.npy"), outputs)

predictions("models/unet_1707955854.3774817.keras", "Data/run1/", "predictions/")
