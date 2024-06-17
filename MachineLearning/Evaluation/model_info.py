from saved_models.saved_models import UNetCustom02CatCyclones

"""
Object keeps track of the models we want to evaluate.
"""
models_info = [
    {
        "Name": "Custom UNet",
        "Output": "Mean 0-2 Category TCs over 10 years",
        "weights": "/nesi/project/uoa03669/ewin313/TropicalCyclone",
        "model": UNetCustom02CatCyclones.load_model(),
        "params": ((55, 105, 1), (11, 13), 1),
    }
]
