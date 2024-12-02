import torch
import sys
import time
import wandb
from MachineLearning.pytorch_dataset import get_pytorch_dataloader
from neuralop.training.callbacks import BasicLoggerCallback
from neuralop.models import FNO2d
from neuralop import Trainer
from neuralop import LpLoss, H1Loss

def train_fno(model_name, data_folder, model_config, training_config):

    model = FNO2d(
        model_config.pop("n_modes_height"),
        model_config.pop("n_modes_width"),          
        **model_config
    )
    
    unique_model_name = '{}_{}'.format(model_name, str(time.time()))
    local_save_path = '../models/{}.keras'.format(unique_model_name)

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
            "param_count": None
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
            "batch_size": config.batch_size,
            "input_description": "[-1, 1] normalized 'genesis_grids'",
            "output_description": "Mean Tropical Cyclone counts over 10 years"
        }
    )


    ## save the train file and unet file so that we can load the model later
    wandb.run.log_code(".", include_fn=lambda p, r: p.endswith("train.py") or p.endswith("unet.py"))

    train_data = get_pytorch_dataloader(data_folder, batch_size=config.batch_size, min_category=training_config["min_category"], max_category=training_config["max_category"])
    test_data = get_pytorch_dataloader(data_folder, dataset="test", batch_size=config.batch_size, min_category=training_config["min_category"], max_category=training_config["max_category"])
    
    device = 'cpu'
    # %%
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=training_config['learning_rate'],
                                 )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # %%
    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    train_loss = l2loss
    eval_losses = {'h1': h1loss, 'l2': l2loss}

    # %%

    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()

    # %%
    # Create the trainer
    trainer = Trainer(model=model, n_epochs=training_config['epoch'],
                      device=device,
                      wandb_log=True,
                      log_test_interval=1,
                      use_distributed=False,
                      log_output=True,
                      verbose=True,
                      callbacks=[BasicLoggerCallback()]
                )

    # %%
    # Actually train the model on our small Darcy-Flow dataset

    trainer.train(train_loader=train_data,
                  test_loaders={32: test_data},
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=False,
                  training_loss=train_loss,
                  eval_losses=eval_losses,
                  save_best="32_l2")

    torch.save(model.state_dict(), local_save_path)