from MachineLearning.DDPMUnet.train import train_ddpm_unet

TRAINING_CONFIG = {
            "Name": "one_batch",
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
    "first_conv_channels": 64,
    "has_attention": [False, False, True, True],
    "interpolation": "bilinear",
    "widths": [8, 16, 32],
    "include_temb": False
}

if __name__ == "__main__":
    train_ddpm_unet(
        'DDPM-Unet',
        './Data/',
        4,
        MODEL_CONFIG,
        TRAINING_CONFIG
    )
