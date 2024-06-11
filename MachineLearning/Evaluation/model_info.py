from saved_models.saved_models import DDPMUNet02CatCyclones, DDPMUNetNoAttention02CatCyclones, UNetCustom02CatCyclones, SongUNet

"""
Object keeps track of the models we want to evaluate.
"""
models_info = [
    {
        "Name": "DDIM Unet",
        "Output": "Mean 0-2 Category TCs over 10 years",
        "model": SongUNet.load_model(),
    },
    {
        "Name": "Custom UNet",
        "Output": "Mean 0-2 Category TCs over 10 years",
        "model": UNetCustom02CatCyclones.load_model(),
    },
    {
        "Name": "DDPM UNet",
        "Output": "Mean 0-2 Category TCs over 10 years",
        "model": DDPMUNet02CatCyclones.load_model(),
    },
    {
        "Name": "DDPM UNet w/o attention",
        "Output": "Mean 0-2 Category TCs over 10 years",
        "model": DDPMUNetNoAttention02CatCyclones.load_model(),
},]
