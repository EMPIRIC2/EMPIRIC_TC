from MachineLearning.UNet.train import train_unet

TRAINING_CONFIG = {
            "Name": "one-batch",
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metric": "mean_absolute_error",
            "learning_rate": 0.003,
            "epoch": 40,
            "batch_size": 32
        }

MODEL_CONFIG = {
    "genesis_size": (112, 224, 1),
    "output_size": (110, 210, 1),
    "kernel_size": (5,5),
    "dropout": False,
    "batch_norm": True,
    "down_filters": [8, 16, 32],
    "up_filters": [32, 16, 8]
}

if __name__ == "__main__":
    train_unet(
        'UNet-Custom',
        '/nesi/project/uoa03669/ewin313/storm_data/v5/',
        4,
        MODEL_CONFIG,
        TRAINING_CONFIG
    )
