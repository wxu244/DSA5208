import numpy as np


class NN:
    def __init__(self, input_dim, hidden_dim, activation='relu', seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation.lower()
        if self.activation_name not in ('relu', 'tanh', 'sigmoid'):
            print("Unsupported activation: choose 'relu', 'tanh' or 'sigmoid'")
            raise ValueError("Unsupported activation: choose 'relu', 'tanh' or 'sigmoid'")
        rng = np.random.RandomState(seed)
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_dim)).astype(np.float64)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float64)
        limit2 = np.sqrt(6.0 / (hidden_dim + 1))
        self.W2 = rng.uniform(-limit2, limit2, size=(hidden_dim, 1)).astype(np.float64)
        self.b2 = np.zeros((1,), dtype=np.float64)

    def activation(self, z):
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        elif self.activation_name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        return None

    def activation_derivative(self, z, a=None):
        if self.activation_name == 'relu':
            dz = (z > 0).astype(np.float64)
            return dz
        elif self.activation_name == 'tanh':
            if a is None:
                a = np.tanh(z)
            return 1.0 - a ** 2
        elif self.activation_name == 'sigmoid':
            if a is None:
                a = 1.0 / (1.0 + np.exp(-z))
            return a * (1.0 - a)
        return None

    def forward_pass(self, x):
        z = x.dot(self.W1) + self.b1
        a = self.activation(z)
        y = a.dot(self.W2) + self.b2
        cache = (x, z, a)
        return y, cache

    def backward(self, y_pred, y_true, cache):
        x, z, a = cache
        dy = (y_pred - y_true)
        dw2 = a.T.dot(dy)
        db2 = np.sum(dy, axis=0)
        dA = dy.dot(self.W2.T)
        dZ = dA * self.activation_derivative(z, a)
        dw1 = x.T.dot(dZ)
        db1 = np.sum(dZ, axis=0)
        grads = {'W1': dw1, 'b1': db1, 'W2': dw2, 'b2': db2}
        return grads

    def apply_gradients(self, grad_avg, lr):
        self.W1 -= lr * grad_avg['W1']
        self.b1 -= lr * grad_avg['b1']
        self.W2 -= lr * grad_avg['W2']
        self.b2 -= lr * grad_avg['b2']
