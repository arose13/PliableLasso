import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin, BaseEstimator


def estimate_intercepts(z, y):
    lm = LinearRegression()
    lm.fit(z, y)

    beta_0 = lm.coef_
    theta_0 = lm.intercept_
    y = y - lm.predict(z)
    return beta_0, theta_0, y


class PliableLasso(BaseEstimator, RegressorMixin):
    def __init__(self, lam=1.0, alpha=0.5):
        self.lam, self.alpha = lam, alpha

        # Model coefs
        self.beta_0 = None
        self.theta_0 = None

    def fit(self, X, Z, y):
        self.beta_0, self.theta_0, y = estimate_intercepts(Z, y)

    def predict(self, X, Z):
        pass
