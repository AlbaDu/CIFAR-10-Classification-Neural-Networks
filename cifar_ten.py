#Import the required libraries
import keras
from keras.datasets import cifar10

def load_dataset():
    """
    ---What it does---
        +Load the Cifar-10 dataset from keras library and transform the labels into categorical features.   
    ---What it returns---
        - train_X
        - test_X
        - train_y
        - test_y 
    """  
    (train_X, train_y),(test_X, test_y) = cifar10.load_data()
	train_y = to_categorical(train_y)
	test_y = to_categorical(test_y)
	return train_X, train_y, test_X, test_y

def data_norm(variable):
    """
    ---What it does---
        + Normalizes the data pixel-color, ranging from 0 (non-color) to 255 (full-color).
    ---What it needs---
        + variable: subset from the dataset.   
    ---What it returns---
        - variable_norm: normalize variable.
    """    
    variable_norm = variable.astype('float32')/ 255.0
    return variable_norm

