import numpy as np
import pickle

class CustomNeuralNetwork:
    def __init__(self, layer_dims, epochs, lr):
        self.layer_dims = layer_dims
        self.epochs = epochs
        self.lr = lr
        self.cost_history = []
        self.params = None

    def initParams(self, layer_dims): 
        np.random.seed(3)
        params = {}
        L = len(layer_dims)

        for l in range(1, L):
            params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        
        self.params = params
        return params

    def loadParams(self, params):
        if not isinstance(params, dict):
            raise ValueError("Los parámetros deben ser un diccionario.")
        
        required_keys = [f"W{l}" for l in range(1, len(self.layer_dims))] + [f"b{l}" for l in range(1, len(self.layer_dims))]
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Falta el parámetro requerido: {key}")

        self.params = params

    def saveParams(self):
        if self.params is None:
            raise ValueError("Los parámetros no han sido inicializados.")
        return self.params

    def forward_prop(self, X):
        A = X
        caches = []
        L = len(self.params) // 2

        for l in range(1, L+1):
            A_prev = A
            Z = np.dot(self.params['W' + str(l)], A_prev) + self.params['b' + str(l)]
            
            if l == L:
                A, activation_cache = self.softmax(Z)
            else:
                A, activation_cache = self.relu(Z)

            linear_cache = (A_prev, self.params['W' + str(l)], self.params['b' + str(l)])
            cache = (linear_cache, activation_cache)
            caches.append(cache)

        return A, caches

    def train(self, X, Y):
        if self.params is None:
            self.initParams(self.layer_dims)
        
        for i in range(self.epochs):
            Y_hat, caches = self.forward_prop(X)
            cost = self.cost_function(Y_hat, Y)
            self.cost_history.append(cost)
            grads = self.backprop(Y_hat, Y, caches)
            self.params = self.update_parameters(self.params, grads, self.lr)

        return self.params, self.cost_history

    def save_model(self, file_path):
        model_data = {
            'layer_dims': self.layer_dims,
            'epochs': self.epochs,
            'lr': self.lr,
            'cost_history': self.cost_history,
            'params': self.params
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            model_data = pickle.load(file)
            self.layer_dims = model_data['layer_dims']
            self.epochs = model_data['epochs']
            self.lr = model_data['lr']
            self.cost_history = model_data['cost_history']
            self.params = model_data['params']
