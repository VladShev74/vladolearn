import numpy as np
import autograd.numpy as npa
from autograd import grad


class MyLinearRegression:
    def __init__(self, regularization: str = "l2", tol: float = 1e-4, alpha: float = 0.5, lr: float = 0.005, seed: int = 42, solver: str = 'gradient'):
        self.regularization = regularization
        self.tol = tol
        self.alpha = alpha
        self.lr = lr
        self.seed = seed
        self.solver = solver

    def fit(self, X, y):
        if self.solver == 'gradient':
            generator = np.random.default_rng(self.seed)
            self.weights = npa.array([generator.normal() for _ in range(X.shape[1])])
            for sample, label in zip(X, y):
                self.sample = npa.array(sample)
                self.label = label
                prediction = npa.dot(self.weights, self.sample)
                loss = self.mse_loss(self.weights, prediction, label)
                gradient = grad(self.mse_loss_grad)(self.weights)
                self.weights = self.weights - self.lr * gradient
        elif self.solver == 'analytical':
            self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        else:
            raise ValueError(f'Wrong solver {self.solver}')

    def mse_loss(self, weights, prediction, y):
        return (prediction - y) ** 2 + self.reg_term(weights)

    def mse_loss_grad(self, weights):
        return (npa.dot(weights, self.sample) - self.label) ** 2 + self.reg_term(weights)

    def reg_term(self, weights):
        if self.regularization == "l1":
            return self.alpha * npa.sum(abs(weights))
        elif self.regularization == "l2":
            return self.alpha * npa.sum(weights ** 2)
        else:
            return 0

    def predict(self, X):
        preds = []
        for sample in X:
            pred = abs(npa.dot(self.weights, sample))
            preds.append(pred)
        return preds
