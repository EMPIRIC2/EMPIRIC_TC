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

def train_unet(model_name, data_folder, data_version, model_config):

    model = UNet(**model_config)

    local_save_path = 'models/{}_{}.keras'.format(model_name, str(time.time()))

    # Start a run, tracking hyperparameters
    wandb.init(
        # set the wandb project where this run will be logged
        project="EMPIRIC2-AI-emulator",

        # track hyperparameters and run metadata with wandb.config
        config={
            "Name": "one-batch",
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metric": "mean_absolute_error",
            "learning_rate": 0.00003,
            "epoch": 40,
            "batch_size": 32
        }
    )

    ## track the model with an artifact
    model_artifact = wandb.Artifact(
        "UNet-custom",
        type="model",
        metadata={
            "save_path": local_save_path,
            "model_config": model_config,
            "param_count": model.count_params()
        }
    )

    wandb.run.log_artifact(model_artifact)

    # [optional] use wandb.config as your config
    config = wandb.config

    ## track the dataset used with an artifact
    data_artifact = wandb.Artifact(
        "processed_data",
        type="dataset",
        metadata={
            "source": "local dataset",
            "data_folder": data_folder,
            "data_version": data_version,
            "batch_size": config.batch_size,
            "input_description": "[-1, 1] normalized 'genesis_grids'",
            "output_description": "mean of 100 TC count grids, 'grid_means'"
        }
    )

    ## save the train file and unet file so that we can load the model later
    wandb.run.log_code(".", include_fn=lambda p, r: p.endswith("train.py") or p.endswith("unet.py"))

    train_data = get_dataset(data_folder, data_version=data_version, batch_size=config.batch_size)
    test_data = get_dataset(data_folder, dataset="test", data_version=data_version)
    
    validation_data = get_dataset(data_folder, dataset="validation", data_version=data_version)

    early_stopping = keras.callbacks.EarlyStopping(patience=5)

    # save best model locally
    checkpoint = keras.callbacks.ModelCheckpoint(local_save_path, save_best_only=True, save_weights_only=True, mode='min',
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
        callbacks=[checkpoint, WandbMetricsLogger(), early_stopping]
    )

    model.evaluate(
        x=test_data,
    )
