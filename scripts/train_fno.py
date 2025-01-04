from MachineLearning.FNO.train import train_fno

TRAINING_CONFIG = {
    "Name": "fno test cat 1 and 2",
    "optimizer": "adam",
    "loss": "mean_squared_error",
    "metric": "mean_absolute_error",
    "learning_rate": 0.001,
    "epoch": 7,
    "batch_size": 32,
    "max_category": 5,
    "min_category": 3,
    "N_100_decades": 1
}

MODEL_CONFIG = {
            "n_modes_height": 16, 
            "n_modes_width": 16,
            "in_channels": 1,
            "hidden_channels": 32,
            "projection_channels": 64,
            "factorization": 'tucker', 
            "rank": 0.42
}

if __name__ == "__main__":
    train_fno(
        'FNO',
        '/users/ewinkelm/data/ewinkelm',
        MODEL_CONFIG,
        TRAINING_CONFIG
    )
