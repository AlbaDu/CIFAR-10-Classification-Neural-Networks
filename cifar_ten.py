#Import the required libraries
import keras
from keras.datasets import cifar10

def data_cat(variable):
    """
    ---What it does---
        +Load the Cifar-10 dataset from keras library and transform the labels into categorical features.   
    ---What it returns---
        - variable_cat: categorized variable. 
    """  
	variable_cat = to_categorical(variable)
	return variable_cat
    
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