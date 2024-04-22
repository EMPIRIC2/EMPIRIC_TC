import tensorflow as tf
from tensorflow import keras
from MachineLearning.dataset import get_dataset
from MachineLearning.UNet.unet import UNet
import time
import numpy as np

genesis_size_default = (55, 105, 1)
movement_size_default = (11, 13)
output_size_default = (110, 210, 1)

def train_unet(data_folder, epochs=10, genesis_size=genesis_size_default, movement_size=movement_size_default, output_size=1):
    train_data = get_dataset(data_folder, data_version=3)
    test_data = get_dataset(data_folder, dataset="test", data_version=3)
    
    validation_data = get_dataset(data_folder, dataset="validation", data_version=3)
    
    model = UNet(genesis_size, movement_size, output_size)

    early_stopping = keras.callbacks.EarlyStopping(patience=5)
    checkpoint = keras.callbacks.ModelCheckpoint('models/unet_mean_{}.keras'.format(str(time.time())), save_best_only=True, save_weights_only=True, mode='min',
                                                 verbose=1)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size"),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    model.fit(
        train_data,
        epochs=epochs,
        verbose=2,
        validation_data=validation_data,
        callbacks=[checkpoint, early_stopping]
    )

    model.evaluate(
        x=test_data,
    )

    model.save('models/unet_mean_{}.keras'.format(str(time.time())))

