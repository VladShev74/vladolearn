import numpy as np
import autograd.numpy as npa
from autograd import grad
from sklearn.metrics import accuracy_score


class MyLogisticRegression:

    def __init__(self, regularization: str = "l2", tol: float = 1e-4, C: float = 5, lr: float = 0.005, seed: int = 42):
        self.regularization = regularization
        self.tol = tol
        self.C = C
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        generator = np.random.default_rng(self.seed)
        self.weights = npa.array([generator.normal() for _ in range(X.shape[1])])
        for sample, label in zip(X, y):
            self.sample = npa.array(sample)
            self.label = label
            prediction = self.sigmoid(self.sample)
            loss = self.log_loss(self.weights, prediction, label)
            gradient = grad(self.log_loss_grad)(self.weights)
            self.weights = self.weights - self.lr * gradient

    def sigmoid(self, sample):
        x = npa.dot(self.weights, sample)
        return 1 / (1 + npa.exp(-x) + 1e-2)

    def sigmoid_grad(self, x):
        return 1 / (1 + npa.exp(-x) + 1e-2)

    def log_loss(self, weights, prediction, y):
        return -(y * npa.log(prediction) + (1-y) * npa.log(1-prediction)) + self.reg_term(weights)

    def log_loss_grad(self, y):
        z = npa.dot(y, self.sample)
        return -(self.label * npa.log(self.sigmoid_grad(z)) + (1-self.label) * npa.log(1 - self.sigmoid_grad(z))) + self.reg_term(y)

    def reg_term(self, weights):
        if self.regularization == "l1":
            return (1 / self.C) * npa.sum(abs(weights))
        elif self.regularization == "l2":
            return (1 / self.C) * npa.sum(weights ** 2)
        else:
            return 0

    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.sigmoid(sample)
            prediction = 0 if prediction < 0.5 else 1
            predictions.append(prediction)
        return predictions

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)
