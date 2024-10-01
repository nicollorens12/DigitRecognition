from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#Pre: The layer's dimension (number of neurons) of each layer in the NN
#Post: Returns a dictonary of params with randomized values
def initial_params(layer_dims): 
    np.random.seed(3)
    params = {}

    L = len(layer_dims)
    rng = np.random.default_rng()

    for l in range(1, L):
        #Generates a matrix with a small weight (in the normal distribution) for each vertex that connects layer l-1 neurons with layer l
        params['W'+str(l)] = rng.standard_normal(size=(layer_dims[l], layer_dims[l-1])) * 0.01 
        params['b'+str(l)] = np.zeros(layer_dims[l],1)
    
    return params