from PlotMapData import plotLatLonGridData
import numpy as np
import matplotlib.pyplot as plt

model_predictions = np.load("../Machine Learning/predictions/model_predictions.npy")
real_outputs = np.load("../Machine Learning/predictions/real_outputs.npy")

print(model_predictions.shape)


#plotLatLonGridData(np.flipud(np.mean(np.sum(real_outputs, axis=3), axis=0)), .5)

for i, prediction in enumerate(model_predictions):
    print("Prediction diff", np.mean(((model_predictions[0] - model_predictions[i])) ** 2))

    plt.imshow(np.sum(model_predictions[0] - model_predictions[i], axis=2))
    plt.show()
    #print(np.mean((model_predictions[i]*1000 - real_outputs[i])**2))
    print("real diff", np.mean((real_outputs[0]*1000 - real_outputs[i]*1000) ** 2))
    plotLatLonGridData(np.flipud(np.sum(real_outputs[i]*1000, axis=2)), .5)
    plotLatLonGridData(np.flipud(np.sum(real_outputs[i]*1000 - real_outputs[0]*1000, axis=2)), .5)