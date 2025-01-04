from tensorflow import keras
from MachineLearning.dataset import get_dataset
from MachineLearning.DDPMUnet.ddpm_unet import build_model
from MachineLearning.PredictionCallback import PredictionCallback
import time
import wandb
from wandb.integration.keras import WandbMetricsLogger

def train_ddpm_unet(
    model_name, 
    data_folder, 
    data_version, 
    model_config, 
    training_config,
):
    
    model = build_model(**model_config)
    unique_model_name = '{}_{}'.format(model_name, str(time.time()))
    local_save_path = 'models/{}.keras'.format(unique_model_name)

    # Start a run, tracking hyperparameters
    wandb.init(
        # set the wandb project where this run will be logged
        project="EMPIRIC2-AI-emulator",

        # track hyperparameters and run metadata with wandb.config
        config=training_config
    )

    ## track the model with an artifact
    model_artifact = wandb.Artifact(
        unique_model_name,
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
            "output_description": "Mean Tropical Cyclone counts over 10 years"
        }
    )

    ## save the train file and unet file so that we can load the model later
    wandb.run.log_code(".", include_fn=lambda p, r: p.endswith("train.py") or p.endswith("unet.py"))

    train_data = get_dataset(
        data_folder,
        data_version=data_version,
        batch_size=config.batch_size,
        min_category=training_config["min_category"],
        max_category=training_config["max_category"],
        N_100_decades=training_config["N_100_decades"]
    )
    
    test_data = get_dataset(
        data_folder, 
        dataset="test", 
        data_version=data_version,
        min_category=training_config["min_category"],
        max_category=training_config["max_category"],
        N_100_decades=training_config["N_100_decades"]
    )

    validation_data = get_dataset(
        data_folder,
        dataset="validation",
        data_version=data_version,
        min_category=training_config["min_category"],
        max_category=training_config["max_category"],
        N_100_decades=training_config["N_100_decades"]
    )

    #validation_example_in, validation_example_out = next(iter(validation_data))

    early_stopping = keras.callbacks.EarlyStopping(patience=5)

    # save best model locally
    checkpoint = keras.callbacks.ModelCheckpoint(local_save_path, save_best_only=True, save_weights_only=True,
                                                 mode='min',
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
