import numpy as np
import pickle

class CustomNeuralNetwork:
    def __init__(self, layer_dims=None, epochs=None, lr=None, pretrained_params=None):
        if pretrained_params:
            self.layer_dims = pretrained_params['layer_dims']
            self.epochs = pretrained_params['epochs']
            self.lr = pretrained_params['lr']
            self.params = pretrained_params['params']
        else:
            self.layer_dims = layer_dims
            self.epochs = epochs
            self.lr = lr
            self.params = self.initParams(layer_dims)
        self.cost_history = []

    def initParams(self, layer_dims): 
        np.random.seed(3)
        params = {}
        L = len(layer_dims)  # Total number of layers

        for l in range(1, L):
            params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        
        return params

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = (Z)
        return A, cache
    
    def relu(self, Z):
        return np.maximum(0, Z), Z
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True)) 
        A = expZ / np.sum(expZ, axis=0, keepdims=True)
        cache = Z
        return A, cache

    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def forward_prop(self, X, params):
        A = X
        caches = []
        L = len(params) // 2

        for l in range(1, L+1):
            A_prev = A
            Z = np.dot(params['W' + str(l)], A_prev) + params['b' + str(l)]
            
            if l == L:
                A, activation_cache = self.softmax(Z)
            else:
                A, activation_cache = self.relu(Z)

            linear_cache = (A_prev, params['W' + str(l)], params['b' + str(l)])
            cache = (linear_cache, activation_cache)
            caches.append(cache)

        return A, caches

    def cost_function(self, A, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def oneLayerBackward(self, dZ, cache, activation):
        linear_cache, activation_cache = cache
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        if activation == "relu":
            Z = activation_cache
            dZ = np.array(dZ, copy=True)
            dZ[Z <= 0] = 0

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db

    def backprop(self, Y_hat, Y, caches):
        grads = {}
        L = len(caches)
        m = Y_hat.shape[1]
        
        Y = Y.reshape(Y_hat.shape)
        
        assert Y_hat.shape == Y.shape
        
        dZL = Y_hat - Y
        current_cache = caches[L-1]
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = self.oneLayerBackward(dZL, current_cache, activation="softmax")

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.oneLayerBackward(grads['dA' + str(l+1)], current_cache, activation="relu")
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l+1)] = dW_temp
            grads['db' + str(l+1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        L = len(parameters) // 2

        for l in range(L):
            parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]

        return parameters

    def train(self, X, Y, layer_dims):
        cost_history = []

        for i in range(self.epochs):
            Y_hat, caches = self.forward_prop(X, self.params)
            cost = self.cost_function(Y_hat, Y)
            cost_history.append(cost)
            grads = self.backprop(Y_hat, Y, caches)
            self.params = self.update_parameters(self.params, grads, self.lr)

        return self.params, cost_history
    
    def predict(self, X, parameters):
        Y_hat, _ = self.forward_prop(X, parameters)
        predictions = np.argmax(Y_hat, axis=0)
        return predictions

    def save_params(self, file_path):
        data = {
            'layer_dims': self.layer_dims,
            'epochs': self.epochs,
            'lr': self.lr,
            'params': self.params
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_params(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

# Example usage:
# nn = CustomNeuralNetwork(layer_dims=[784, 128, 10], epochs=1000, lr=0.01)
# nn.train(X_train, Y_train, nn.layer_dims)
# nn.save_params('model_params.pkl')
# pretrained_params = CustomNeuralNetwork.load_params('model_params.pkl')
# nn_pretrained = CustomNeuralNetwork(pretrained_params=pretrained_params)
