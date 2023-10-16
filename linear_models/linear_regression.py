import autograd.numpy as npa
import numpy as np
from autograd import grad

from model_carcass import Model


class MyLinearRegression(Model):
    def __init__(self,
                 regularization: str = "l2",
                 tol: float = 1e-4,
                 alpha: float = 0.5,
                 lr: float = 0.005,
                 seed: int = 42,
                 solver: str = 'gradient'):
        self.regularization = regularization
        self.tol = tol
        self.alpha = alpha
        self.lr = lr
        self.seed = seed
        self.solver = solver

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits model on training data with two possible solvers (analytical or gradient).

        Parameters
        __________
            X : np.ndarray
                Input array with samples.
            y : np.ndarray
                Array of binary target values.

        Returns
        _______
            None
        """
        if self.solver == 'gradient':
            generator = np.random.default_rng(self.seed)
            self.weights = npa.array([generator.normal() for _ in range(X.shape[1])])
            for sample, label in zip(X, y):
                self.sample = npa.array(sample)
                self.label = label
                prediction = npa.dot(self.weights, self.sample)
                loss = self.mse_loss(self.weights, prediction, label)
                gradient = grad(self.__mse_loss_grad)(self.weights)
                self.weights = self.weights - self.lr * gradient
        elif self.solver == 'analytical':
            self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        else:
            raise ValueError(f'Wrong solver {self.solver}')

    def mse_loss(self, weights: np.ndarray, prediction: float, y: float) -> float:
        """
        Calculates MSE loss using weights, prediction and one target value.

        Parameters
        __________
            weights : np.ndarray
                Array of model weights.
            prediction : float
                Predicted value.
            y : float
                Float target value.

        Returns
        _______
            float
                MSE loss.
        """
        return (prediction - y) ** 2 + self.__reg_term(weights)

    def __mse_loss_grad(self, weights: np.ndarray) -> float:
        """
        Calculates MSE loss using all manual calculations for gradient update.

        Parameters
        __________
            weights : np.ndarray
                Array of model weights.

        Returns
        _______
            float
                MSE loss for gradient.
        """
        return (npa.dot(weights, self.sample) - self.label) ** 2 + self.__reg_term(weights)

    def __reg_term(self, weights: np.ndarray) -> float:
        """
        Defines the regularization term (L1 or L2) used for loss calculation.

        Parameters
        __________
            weights : np.ndarray
                Array of model weights.

        Returns
        _______
            float
                Regularization term multiplied by alpha value.
        """
        if self.regularization == "l1":
            return self.alpha * npa.sum(abs(weights))
        elif self.regularization == "l2":
            return self.alpha * npa.sum(weights ** 2)
        else:
            return 0.0

    def predict(self, X: np.ndarray) -> list:
        """
        Writes all predictions in the list and returns them.

        Parameters
        __________
            X : np.ndarray
                Array of samples.

        Returns
        _______
            list
                List of predicted values.
        """
        preds = []
        for sample in X:
            pred = abs(npa.dot(self.weights, sample))
            preds.append(pred)
        return preds
