from MachineLearning.SongUNet.train import train_songunet

TRAINING_CONFIG = {
            "Name": "Song UNet",
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metric": "mean_absolute_error",
            "learning_rate": 0.003,
            "epoch": 40,
            "batch_size": 32
        }

MODEL_CONFIG = {
    "input_size": (112, 224, 1),
    "output_size": (110, 210, 1),
    "widths": [32, 64, 128],
    "block_depth": 2
}

if __name__ == "__main__":
    train_songunet(
        'SongUNet',
        '/users/ewinkelm/data/ewinkelm',
        4,
        MODEL_CONFIG,
        TRAINING_CONFIG
    )
