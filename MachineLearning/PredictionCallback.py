import keras
import matplotlib.pyplot as plt

class PredictionCallback(keras.callbacks.Callback):

  def __init__(self, x, y):
    super().__init__()
    self.x = x
    self.y = y

  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict(self.x)

    plt.imshow(y_pred[0])
    plt.show()