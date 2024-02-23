from PlotMapData import plotLatLonGridData
import numpy as np
import matplotlib.pyplot as plt

model_predictions = np.load("../MachineLearning/predictions/model_predictions1.npy")
real_outputs = np.load("../MachineLearning/predictions/real_outputs1.npy")

print(model_predictions.shape)


#plotLatLonGridData(np.flipud(np.mean(np.sum(real_outputs, axis=3), axis=0)), .5)

for i, prediction in enumerate(model_predictions):
    print("Prediction diff", np.mean(((model_predictions[0] - model_predictions[i])) ** 2))
    plotLatLonGridData(np.sum(prediction, axis=2), .5)

    #plt.imshow(np.sum(prediction, axis=2))
    #plt.show()
    #plt.imshow(np.sum(model_predictions[0] - model_predictions[i], axis=2))
    #plt.show()

    #plt.imshow(np.sum(prediction - real_outputs[i], axis=2))
    #plt.show()
    #print(np.mean((model_predictions[i]*1000 - real_outputs[i])**2))
    print("real diff", np.mean((real_outputs[0] - real_outputs[i]) ** 2))
    plotLatLonGridData(np.sum(real_outputs[i], axis=2), .5)
    #plotLatLonGridData(np.sum(real_outputs[i] - real_outputs[0], axis=2), .5)