from typing import Optional

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate: float, epochs: int = 10000):
        self.weights: np.array = None
        self.bias: Optional[float] = None

        self.learning_rate = learning_rate
        self.epochs = epochs

    def calculate_lasso_loss(self, y: np.array, y_pred: np.array) -> float:
        return np.sum(np.abs(y - y_pred))

    def fit(self, X: np.array, y: np.array):
        y = y.reshape(-1, 1)
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=(X.shape[1], 1))

        self.bias = 0

        for epoch in range(self.epochs):

            error = self.calculate_lasso_loss(y, self.predict(X))

            print(error)

            n = len(X)

            sign = np.sign(y - self.predict(X))
            dw = -X.T @ sign / len(X) # L1 Derived (Weights)
            db = -np.sum(sign) / len(X) # L1 Derived (Bias)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x: np.array):
        return x @ self.weights + self.bias


lr = LinearRegression(learning_rate=0.001)

lr.fit(
    np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]),
    np.array([1, 2, 3, 4, 5])
)

result = lr.predict(np.array([8, 8, 8]))

# Deveria se 13
print("Resultado: ", result)