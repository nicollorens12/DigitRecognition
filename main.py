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
        #For visualization is kind of and Adjacency Matrix that indicates with it's index the conection (edge) and it's weight
        params['W'+str(l)] = rng.standard_normal(size=(layer_dims[l], layer_dims[l-1])) * 0.01 
        params['b'+str(l)] = np.zeros(layer_dims[l],1)
    
    return params

# Other activation functions can be used 
# Z (linear hypothesis) - Z = W*X + b , 
# W - weight matrix, b- bias vector, X- Input 

def sigmoid(Z):
    A = 1/(1+np.exp(np.dot(-1, Z)))
    cache = (Z)

    return A, cache

#Takes traning data and parameters
#Generates output for one layer and then it will feed that output to the next layer and so on.
def forward_prop(X, params):
    A = X # input to first layer i.e. training data
    caches = []
    L = len(params)//2 #As each layer has W matrix and b array, L is the number of Layers of our network

    for l in range(1,L+1):
        A_prev = A

        # Linear combination of weights, inputs and biases to get the resulting activation values (before sigmoid squeezification)
        Z = np.dot(params['W'+str(l)], A_prev) + params['b'+str(l)] 

        # Storing the linear cache
        linear_cache = (A_prev, params['W'+str(l)], params['b'+str(l)]) 

        # Applying sigmoid on linear hypothesis
        A, activation_cache = sigmoid(Z) #activation_cache is Z before sigmoid

        cache = (linear_cache, activation_cache) 
        caches.append(cache)
    
    return A, caches

#Takes the activation values of the last layer of our NN and the real values (what A should be)
#A is a value between 0 and 1 (thanks to the sigmoid function) but Y is binary (0 or 1) that indicate the real class 

def cost_function(A, Y):
    m = Y.shape[1] #m is the number of training examples, this makes cost independent from dataset size

    #Cost is the average (that why is divided by m)
    #Uses log so that the further the A value of a certein neuron is compared what it should be, the more penalized it is
    cost = (-1/m)*(np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), 1-Y.T)) 

    return cost

