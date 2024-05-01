import tensorflow as tf
from tensorflow import keras
from MachineLearning.dataset import get_dataset
from MachineLearning.UNet.unet import UNet
from keras.utils.layer_utils import count_params
import time
import numpy as np
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

genesis_size_default = (55, 105, 1)
movement_size_default = (11, 13)
output_size_default = (110, 210, 1)

def train_unet(data_folder, epochs=10, genesis_size=genesis_size_default, movement_size=movement_size_default, output_size=1):
    
    # Start a run, tracking hyperparameters
    wandb.init(
        # set the wandb project where this run will be logged
        project="EMPIRIC2-AI-emulator",

        # track hyperparameters and run metadata with wandb.config
        config={
            "name": "one-batch",
            "batch_normalization": True,
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metric": "mean_absolute_error",
            "learning_rate": 0.00003,
            "epoch": 40,
            "batch_size": 32
        }
    )

    # [optional] use wandb.config as your config
    config = wandb.config

    train_data = get_dataset(data_folder, data_version=3, batch_size=config.batch_size)
    test_data = get_dataset(data_folder, dataset="test", data_version=3)
    
    validation_data = get_dataset(data_folder, dataset="validation", data_version=3)
    
    model = UNet(genesis_size, movement_size, output_size)
    
    print("Number of parameters: {}".format(model.count_params()))

    early_stopping = keras.callbacks.EarlyStopping(patience=5)
    checkpoint = keras.callbacks.ModelCheckpoint('models/unet_mean_{}.keras'.format(str(time.time())), save_best_only=True, save_weights_only=True, mode='min',
                                                 verbose=1)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size"),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    model.fit(
        train_data,
        epochs=config.epoch,
        verbose=2,
        validation_data=validation_data,
        callbacks=[WandbModelCheckpoint("models"), WandbMetricsLogger, early_stopping]
    )

    model.evaluate(
        x=test_data,
    )
