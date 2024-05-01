from MachineLearning.ConvProbPredictor.predict import make_site_predictions
print("making site preds")
make_site_predictions("/nesi/project/uoa03669/ewin313/storm_data/v2/", "/nesi/project/uoa03669/ewin313/TropicalCycloneAI/models/site_prob_discrete_1712630188.1991909.weights.h5", "/nesi/project/uoa03669/ewin313/TropicalCycloneAI/MachineLearning/ConvProbPredictor/predictions", 9)


