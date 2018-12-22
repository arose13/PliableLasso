import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.base import RegressorMixin, BaseEstimator


class PliableLasso(BaseEstimator, RegressorMixin):
    def __init__(self, lam=1.0, alpha=0.5):
        self.lam, self.alpha = lam, alpha
