"""Implementation of a Regression Model that Fits Data to a Sum of Sine and Polynomial.

        This file implements a regression model that fits data to a sum of a polynomial and a sine function using
        gradient descent. The loss function used is the mean squarred error (MSE).
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"""Internal class that implements the regression part of the model.
"""
class Regression:
    """Initializes the regression model with the number of iterations and learning rate.
        
        Args:
            iterations (int): Number of iterations for gradient descent.
            lr (float): Learning rate for gradient descent.
    """
    def __init__(self, iterations = 1000, lr = 0.01):
        self.iterations = iterations
        self.lr = lr
        
        self.poly_weights = None # Will be initialized to fit the degree of the polynomial given.
        self.bias = 0.0          # Bias term
        self.sin_amp = 1     # Amplitude of the sine
        self.sin_ang_freq = 1  # Angular frequency of the sine
        self.sin_phase = 0

    """Fits the regression model to the data using gradient descent.
        Args:
            X (array-like): Input features.
            Y (array-like): Target values.

        Returns:
            self: The fitted regression model.
    """
    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)
        _ , degree = X.shape

        #Initialize weights to random values
        self.poly_weights = np.zeros(X.shape[1])

        for iter in range(self.iterations):
            # Predict the outputs based on these inputs
            Y_pred = self.predict(X)

            # Calculate the gradient and assign
            diff = np.array([a - b for a,b in zip(Y_pred, Y)]) # I don't trust numpy
            poly_grad = np.array([2*np.mean([a*b for a,b in zip(X.T[i],diff)]) for i in range(degree)])
            bias_grad = 2 * np.mean(diff)

            # Feature maps for the sine and cosine function (cannot be done before due to the angular
            #  frequency being different on each iteration)
            sin_term = np.sin((self.sin_ang_freq * X) + self.sin_phase)
            cos_term = np.cos((self.sin_ang_freq * X) + self.sin_phase)

            sin_amp_grad = 2 * np.mean([a*b for a,b in zip(sin_term, diff)])
            sin_ang_freq_grad = (2*self.sin_amp) * np.mean([a*b*c for a,b,c in zip(diff, cos_term, X.T[0])])
            sin_phase_grad = (2*self.sin_amp) * np.mean([a*b for a,b in zip(diff, cos_term)])

            # Decrement the weights accordingly
            self.poly_weights -= self.lr * poly_grad
            self.bias -= self.lr * bias_grad
            self.sin_amp -= self.lr * sin_amp_grad
            self.sin_ang_freq -= self.lr * sin_ang_freq_grad
            self.sin_phase -= self.lr * sin_phase_grad

        return self
    
    """Predicts the target values based on the input features.
        Args:
            X (array-like): Input features.

        Returns:
            array: Predicted values.
    """
    def predict(self, X):
        X = np.asarray(X)

        return np.dot(X, self.poly_weights) + self.bias + \
                (self.sin_amp * np.sin((self.sin_ang_freq * X.T[0]) + self.sin_phase))

"""Outward facing API for the module. Creates the model and is responsible for fitting and predicting data.
        This class automatically reshapes the input arrays.
"""
class SineWithPolyModel():
    """Initializes the model with the given degree of polynomial, number of iterations for gradient descent,
        and learning rate.
        
        Args:
            degree (int): Degree of the polynomial to fit.
            iterations (int): Number of iterations for gradient descent.
            lr (float): Learning rate for gradient descent.
    """
    def __init__(self, degree = 1, iterations = 1000, lr = 0.01):
        self.__scaler = StandardScaler()
        self.__poly_feat = PolynomialFeatures(degree, include_bias=False)
        self.__regressor = Regression(iterations = iterations, lr = lr)
        self.__model = make_pipeline(self.__scaler, self.__poly_feat, self.__regressor)

    """Wrapper for the fitting of the model.
        Args:
            X (array-like): Input features.
            Y (array-like): Target values.
        
        Returns:
            self: The fitted model.
    """
    def fit(self, X, Y):

        self.__model.fit(X,Y)

        return self
    
    """Wrapper for predicting targets based on data given
        Args:
            X (array-like): Input features.
        
        Returns:
            array: Predicted values.
    """
    def predict(self, X):
        return self.__model.predict(X)