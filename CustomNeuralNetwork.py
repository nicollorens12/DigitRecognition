import numpy as np

class CustomNeuralNetwork:
    def __init__(self, layer_dims, epochs, lr):
        self.layer_dims = layer_dims
        self.epochs = epochs
        self.lr = lr
        self.cost_history = []

    #Pre: The layer's dimension (number of neurons) of each layer in the NN
    #Post: Returns a dictonary of params with randomized values
    def initParams(self, layer_dims): 
        np.random.seed(3)
        params = {}
        L = len(layer_dims)  # Total number of layers

        for l in range(1, L):
            # The random values are multiplied by np.sqrt(2 / layer_dims[l-1]) to avoid vanishing/exploding gradients
            # as *0.01 was vanishing the gradients
            params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            #params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        
        return params


    # Other activation functions can be used 
    # Z (linear hypothesis) - Z = W*X + b , 
    # W - weight matrix, b- bias vector, X- Input 
    def sigmoid(self,Z):
        A = 1 / (1 + np.exp(-Z))
        cache = (Z)
        
        return A, cache
    
    #ReLU is used in hidden layers of the network
    #It is used to introduce non-linearity in the network
    #It is computationally less expensive than sigmoid and tanh
    def relu(self, Z):
        return np.maximum(0, Z), Z
    
    #Softmax function is used for multi-class classification problems
    #It is used in the output layer of the network to get the probabilities of each class as a total of 1
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True)) 
        A = expZ / np.sum(expZ, axis=0, keepdims=True)
        cache = Z  # Almacena Z como cache para la retropropagación
        return A, cache

    
    
    #Relu backward is used in backpropagation
    #It is used to calculate the gradients of the weights and biases
    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)  # Convertir dA en un array en lugar de una referencia
        dZ[Z <= 0] = 0  # Gradiente es 0 para Z <= 0
        return dZ

    #Takes traning data and parameters
    #Generates output for one layer and then it will feed that output to the next layer and so on.
    #Cache is needed for the future step 'backpropagation'
    def forward_prop(self, X, params):
        A = X  # input to the first layer
        caches = []
        L = len(params) // 2  # Number of layers (W and b for each layer)

        for l in range(1, L+1):
            A_prev = A
            Z = np.dot(params['W' + str(l)], A_prev) + params['b' + str(l)]
            
            if l == L:  # Last layer, use softmax
                A, activation_cache = self.softmax(Z)
            else:  # Hidden layers, use ReLU
                A, activation_cache = self.relu(Z)

            # Save the caches for backpropagation
            linear_cache = (A_prev, params['W' + str(l)], params['b' + str(l)])
            cache = (linear_cache, activation_cache)
            caches.append(cache)

        return A, caches


    #Takes the activation values of the last layer of our NN and the real values (what A should be)
    #A is a value between 0 and 1 (thanks to the sigmoid function) but Y is binary (0 or 1) that indicate the real class 
    def cost_function(self, A, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m  # Añadir epsilon para evitar log(0)
        return cost



    def oneLayerBackward(self, dZ, cache, activation):
        linear_cache, activation_cache = cache
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        if activation == "relu":
            Z = activation_cache
            dZ = np.array(dZ, copy=True)
            dZ[Z <= 0] = 0  # ReLU derivative

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db



    #Backprop is the process of updating the weights and biases of the network to minimize the cost function
    #Pre: The activation values of the last layer of our NN, the real values and the caches
    #Post: Returns the gradients of the weights and biases
    def backprop(self, Y_hat, Y, caches):
        grads = {}
        L = len(caches)  # Number of layers
        m = Y_hat.shape[1]
        
        Y = Y.reshape(Y_hat.shape)
        
        assert Y_hat.shape == Y.shape # Asegurar que las dimensiones son iguales
        
        # Calculate dZL for softmax (output layer)
        dZL = Y_hat - Y
        current_cache = caches[L-1]
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = self.oneLayerBackward(dZL, current_cache, activation="softmax")


        # Calcular los gradientes para la última capa (usando sigmoid en este ejemplo)
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.oneLayerBackward(grads['dA' + str(l+1)], current_cache, activation="relu")
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l+1)] = dW_temp
            grads['db' + str(l+1)] = db_temp

        return grads





    #Pre: All grads have been already computed
    #Post: Updates the parameters of the network
    #Learning rate is a hyperparameter that indicates how much the weights and biases will be updated
    # (the higher the learning rate, the faster the network will learn but it can also overshoot the minimum)
    # (the lower the learning rate, the slower the network will learn but it can also get stuck in a local minimum)
    def update_parameters(self,parameters, grads, learning_rate):
        L = len(parameters) // 2

        for l in range(L):
            parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]


        return parameters

    #Pre: The training data, the real values, the layer's dimension (number of neurons) of each layer in the NN, the number of epochs and the learning rate
    #Post: Returns the parameters of the network and the cost history
    # Epochs is a hyperparameter that indicates how many times the network will see the training data
    # (the more epochs, the more the network will learn but it can also overfit the data)
    def train(self, X, Y, layer_dims):
        params = self.initParams(layer_dims)
        cost_history = []

        for i in range(self.epochs):
            Y_hat, caches = self.forward_prop(X, params)
            cost = self.cost_function(Y_hat, Y)
            cost_history.append(cost)
            grads = self.backprop(Y_hat, Y, caches)
            params = self.update_parameters(params, grads, self.lr)

        return params, cost_history
    
    def predict(self, X, parameters):
        Y_hat, _ = self.forward_prop(X, parameters)
        predictions = np.argmax(Y_hat, axis=0)  # Para clasificación multi-clase
        return predictions

class CustomNeuralNetworkV2:
    def __init__(self, layer_dims, epochs, lr):
        self.layer_dims = layer_dims
        self.epochs = epochs
        self.lr = lr
        self.cost_history = []

    #Pre: The layer's dimension (number of neurons) of each layer in the NN
    #Post: Returns a dictonary of params with randomized values
    def initParams(self, layer_dims): 
        np.random.seed(3)
        params = {}
        L = len(layer_dims)  # Total number of layers

        for l in range(1, L):
            # The random values are multiplied by np.sqrt(2 / layer_dims[l-1]) to avoid vanishing/exploding gradients
            # as *0.01 was vanishing the gradients
            params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            #params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        
        return params


    # Other activation functions can be used 
    # Z (linear hypothesis) - Z = W*X + b , 
    # W - weight matrix, b- bias vector, X- Input 
    def sigmoid(self,Z):
        A = 1 / (1 + np.exp(-Z))
        cache = (Z)
        
        return A, cache
    
    #ReLU is used in hidden layers of the network
    #It is used to introduce non-linearity in the network
    #It is computationally less expensive than sigmoid and tanh
    def relu(self, Z):
        return np.maximum(0, Z), Z
    
    #Softmax function is used for multi-class classification problems
    #It is used in the output layer of the network to get the probabilities of each class as a total of 1
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True)) 
        A = expZ / np.sum(expZ, axis=0, keepdims=True)
        cache = Z  # Almacena Z como cache para la retropropagación
        return A, cache

    
    
    #Relu backward is used in backpropagation
    #It is used to calculate the gradients of the weights and biases
    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)  # Convertir dA en un array en lugar de una referencia
        dZ[Z <= 0] = 0  # Gradiente es 0 para Z <= 0
        return dZ

    #Takes traning data and parameters
    #Generates output for one layer and then it will feed that output to the next layer and so on.
    #Cache is needed for the future step 'backpropagation'
    def forward_prop(self, X, params):
        A = X  # input to the first layer
        caches = []
        L = len(params) // 2  # Number of layers (W and b for each layer)

        for l in range(1, L+1):
            A_prev = A
            Z = np.dot(params['W' + str(l)], A_prev) + params['b' + str(l)]
            
            if l == L:  # Last layer, use softmax
                A, activation_cache = self.softmax(Z)
            else:  # Hidden layers, use ReLU
                A, activation_cache = self.relu(Z)

            # Save the caches for backpropagation
            linear_cache = (A_prev, params['W' + str(l)], params['b' + str(l)])
            cache = (linear_cache, activation_cache)
            caches.append(cache)

        return A, caches


    #Takes the activation values of the last layer of our NN and the real values (what A should be)
    #A is a value between 0 and 1 (thanks to the sigmoid function) but Y is binary (0 or 1) that indicate the real class 
    def cost_function(self, A, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m  # Añadir epsilon para evitar log(0)
        return cost



    def oneLayerBackward(self, dZ, cache, activation):
        linear_cache, activation_cache = cache
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        if activation == "relu":
            Z = activation_cache
            dZ = np.array(dZ, copy=True)
            dZ[Z <= 0] = 0  # ReLU derivative

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db



    #Backprop is the process of updating the weights and biases of the network to minimize the cost function
    #Pre: The activation values of the last layer of our NN, the real values and the caches
    #Post: Returns the gradients of the weights and biases
    def backprop(self, Y_hat, Y, caches):
        grads = {}
        L = len(caches)  # Number of layers
        m = Y_hat.shape[1]
        
        Y = Y.reshape(Y_hat.shape)
        
        assert Y_hat.shape == Y.shape # Asegurar que las dimensiones son iguales
        
        # Calculate dZL for softmax (output layer)
        dZL = Y_hat - Y
        current_cache = caches[L-1]
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = self.oneLayerBackward(dZL, current_cache, activation="softmax")


        # Calcular los gradientes para la última capa (usando sigmoid en este ejemplo)
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.oneLayerBackward(grads['dA' + str(l+1)], current_cache, activation="relu")
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l+1)] = dW_temp
            grads['db' + str(l+1)] = db_temp

        return grads





    #Pre: All grads have been already computed
    #Post: Updates the parameters of the network
    #Learning rate is a hyperparameter that indicates how much the weights and biases will be updated
    # (the higher the learning rate, the faster the network will learn but it can also overshoot the minimum)
    # (the lower the learning rate, the slower the network will learn but it can also get stuck in a local minimum)
    def update_parameters(self,parameters, grads, learning_rate):
        L = len(parameters) // 2

        for l in range(L):
            parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]


        return parameters

    #Pre: The training data, the real values, the layer's dimension (number of neurons) of each layer in the NN, the number of epochs and the learning rate
    #Post: Returns the parameters of the network and the cost history
    # Epochs is a hyperparameter that indicates how many times the network will see the training data
    # (the more epochs, the more the network will learn but it can also overfit the data)
    def train(self, X, Y, layer_dims):
        params = self.initParams(layer_dims)
        cost_history = []

        for i in range(self.epochs):
            Y_hat, caches = self.forward_prop(X, params)
            cost = self.cost_function(Y_hat, Y)
            cost_history.append(cost)
            grads = self.backprop(Y_hat, Y, caches)
            params = self.update_parameters(params, grads, self.lr)

        return params, cost_history
    
    def predict(self, X, parameters):
        Y_hat, _ = self.forward_prop(X, parameters)
        predictions = np.argmax(Y_hat, axis=0)  # Para clasificación multi-clase
        return predictions