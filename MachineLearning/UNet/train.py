import tensorflow as tf
from tensorflow import keras
from MachineLearning.dataset import get_dataset
from unet import UNet
import time
import tensorflow_probability as tfp
import numpy as np

genesis_size_default = (55, 105, 6)
movement_size_default = (11, 13)
output_size_default = (110, 210, 6)

@keras.saving.register_keras_serializable(package="ProbabilityLayers", name="NegLogLik")
def MeanSquaredWithQuantiles(y_true, y_pred):
    q = np.array([0, 0.001, 0.01, 0.1, 0.3, .5, .7, .9, .99, .999, 1]) * 100
    # calculate  the quantile or percentile value for each, tf.linalg.quantile()
    # and take the mean squared error of all the percentiles -> then take the mean squared error between the images and add together
    # this is custom loss to try and see if we can generate better extremes with the UNet
    quant_true = tfp.stats.percentile(y_true, q)
    quant_pred = tfp.stats.percentile(y_pred, q)

    return keras.losses.mean_squared_error(quant_true, quant_pred) + keras.losses.mean_squared_error(y_true, y_pred)

def train_unet(data_folder, epochs=50,genesis_size=genesis_size_default, output_size=output_size_default):
 
    train_data = get_dataset(data_folder)
    test_data = get_dataset(data_folder)
    model = UNet(genesis_size)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=MeanSquaredWithQuantiles,
        metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    )

    model.fit(
        x=train_data.skip(2),
        epochs=epochs,
        verbose=2,
        validation_data=train_data.take(2)
    )

    model.evaluate(
        x=test_data,
    )

    model.save('models/unet_{}.keras'.format(str(time.time())))
keras.backend.clear_session()
train_unet('Data/run1/')

