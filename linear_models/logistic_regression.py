import autograd.numpy as npa
import numpy as np
from autograd import grad
from sklearn.metrics import accuracy_score

from model_carcass import Model


class MyLogisticRegression(Model):

    def __init__(self,
                 regularization: str = "l2",
                 tol: float = 1e-4,
                 C: float = 5,
                 lr: float = 0.005,
                 seed: int = 42):
        self.regularization = regularization
        self.tol = tol
        self.C = C
        self.lr = lr
        self.seed = seed

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
        generator = np.random.default_rng(self.seed)
        self.weights = npa.array([generator.normal() for _ in range(X.shape[1])])
        for sample, label in zip(X, y):
            self.sample = npa.array(sample)
            self.label = label
            prediction = self.sigmoid(self.sample)
            loss = self.log_loss(self.weights, prediction, label)
            gradient = grad(self.__log_loss_grad)(self.weights)
            self.weights = self.weights - self.lr * gradient

    def sigmoid(self, sample: np.ndarray) -> float:
        """
        Calculates sigmoid function on weights and sample.

        Parameters
        __________
            sample : np.ndarray
                Vector of sample values

        Returns
        _______
            float
                Result of sigmoid function calculation.
        """
        x = npa.dot(self.weights, sample)
        return 1 / (1 + npa.exp(-x) + 1e-2)

    def __sigmoid_grad(self, x: float) -> float:
        """
        Calculates sigmoid function on dot product of sample and weights for gradient update.

        Parameters
        __________
            x : float
                Dot product of sample and weights.

        Returns
        _______
            float
                Result of sigmoid function calculation.
        """
        return 1 / (1 + npa.exp(-x) + 1e-2)

    def log_loss(self, weights: np.ndarray, prediction: float, y: int) -> float:
        """
        Calculates negative log loss on given data.

        Parameters
        __________
            weights : np.ndarray
                Model weights, used in regularization.
            prediction : float
                Predicted value, result of sigmoid function.
            y : int
                Target value, can be only 0 or 1.

        Returns
        _______
            float
                Result of negative log loss calculation.
        """
        return -(y * npa.log(prediction) + (1-y) * npa.log(1-prediction)) + self.__reg_term(weights)

    def __log_loss_grad(self, y: npa.numpy_boxes.ArrayBox) -> float:
        """
        Calculates negative log loss on given data for gradient update.

        Parameters
        __________
            y : autograd.numpy.numpy_boxes.ArrayBox
                Array of model weights.

        Returns
        _______
            float
                Result of negative log loss calculation.
        """
        z = npa.dot(y, self.sample)
        return -(self.label * npa.log(self.__sigmoid_grad(z)) + (1-self.label) * npa.log(1 - self.__sigmoid_grad(z))) + self.__reg_term(y)

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
                Regularization term multiplied by multiplier value (1/C).
        """
        if self.regularization == "l1":
            return (1 / self.C) * npa.sum(abs(weights))
        elif self.regularization == "l2":
            return (1 / self.C) * npa.sum(weights ** 2)
        else:
            return 0.0

    def predict(self, X: np.ndarray) -> list:
        """
        Writes all predictions in the list and returns them.

        Parameters
        __________
            X : np.ndarray
                Input array with samples.

        Returns
        _______
            list
                List of predicted values.
        """
        predictions = []
        for sample in X:
            prediction = self.sigmoid(sample)
            prediction = 0 if prediction < 0.5 else 1
            predictions.append(prediction)
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Shows accuracy score on the list of predicted values.

        Parameters
        __________
            X : np.ndarray
                Input array with samples.
            y : np.ndarray
                Array of binary target values.

        Returns
        _______
            float
                Accuracy score value.
        """
        preds = self.predict(X)
        return accuracy_score(y, preds)
