#Import the required libraries
import keras
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def target_cat(label):
    """
    ---What it does---
        + Transforms the labels into categorical features.
    ---What it needs---
        +label.
    ---What it returns---
        + label_cat: categorized labels.
    """
    label_cat = to_categorical(label)
    return label_cat

def data_norm(variable):
    """
    ---What it does---
        + Normalizes the data pixel-color, ranging from 0 to 255 (non-color to full-color)
    ---What it needs---
        + variable.
    ---What it returns---
        + variable_norm: normalized variable.
    """
    variable_norm = variable.astype("float32")/255.0
    return variable_norm

def acc_plot(history_):
    """
    ---What it does---
        + It plots the training and validation accuracy based on epochs number.
    ---What it needs---
        + history_: information stored during the model fitting process
    """

    history_dict = history.history
    acc = history.history_["acc"]
    val_acc = history.history_["val_acc"]
    epochs = range(len(acc))

    plt.plot(epochs, acc, "bo", label = "Training acc")
    plt.plot(epochs, val_acc, "b", label = "Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.show()

