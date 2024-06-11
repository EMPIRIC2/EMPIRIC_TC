from MachineLearning.DDPMUnet.train import train_ddpm_unet

TRAINING_CONFIG = {
            "Name": "No Attention Ablation",
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metric": "mean_absolute_error",
            "learning_rate": 0.003,
            "epoch": 40,
            "batch_size": 32
        }

MODEL_CONFIG = {
    "img_size": (112, 224, 1),
    "output_size": (110, 210, 1),
    "has_attention": [False, False, True],
    "interpolation": "bilinear",
    "widths": [16, 32, 64],
    "include_temb": False
}

if __name__ == "__main__":
    train_ddpm_unet(
        'DDPM-Unet',
        '/nesi/project/uoa03669/ewin313/storm_data/v5/',
        4,
        MODEL_CONFIG,
        TRAINING_CONFIG
    )
