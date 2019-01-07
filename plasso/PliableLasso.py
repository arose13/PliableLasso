import numpy as np
import pandas as pd
import numpy.linalg as la
import scipy.optimize as opt
from tqdm import trange
from sklearn.exceptions import NotFittedError
from sklearn.base import RegressorMixin, BaseEstimator
from .helpers import *


# noinspection PyPep8Naming
class PliableLasso(BaseEstimator, RegressorMixin):
    """
    Pliable Lasso https://arxiv.org/pdf/1712.00484.pdf
    """
    def __init__(self, lam=1.0, alpha=0.5, max_iter=100):
        self.lam, self.alpha = lam, alpha
        self.max_iter = max_iter

        # Model coefs
        self.beta_0 = None
        self.theta_0 = None
        self.beta = None
        self.theta = None

        # Metrics
        self.history = []

    def fit(self, X, Z, y):
        self.history = []

        # NOTE: wtf this is an O(nk)
        alpha, lam = self.alpha, self.lam  # So I don't have to keep writing self
        n, p = X.shape
        k = Z.shape[1]

        # Step 1: Compute beta_0 and theta_0
        self.beta, self.theta = np.zeros(p), np.zeros((p, k))
        self.beta_0, self.theta_0, y = estimate_intercepts(Z, y)

        self.history.append(
            j(self.beta_0, self.theta_0, self.beta, self.theta, X, Z, y, alpha, lam)
        )

        # Step 2: Update coefficients
        for i in trange(self.max_iter):

            # Iterate through all p features
            for pi in range(p):
                r_minus_j = y - partial_model(self.beta, self.theta, X, Z, pi)

                # Condition 17
                x_j_t_dot_r_minus_j_over_n = (v2a(X[:, pi]).T @ r_minus_j) / n
                cond_17a = abs(x_j_t_dot_r_minus_j_over_n)
                cond_17b = soft_thres(compute_w_j(X, Z, pi).T @ r_minus_j / n, alpha * lam)
                cond_17b = la.norm(cond_17b, 2)
                if cond_17a <= (1 - alpha) * lam and cond_17b <= 2 * (1 - alpha) * lam:
                    # Step 2.a: beta_j_hat == 0 and theta_j_hat == 0
                    continue  # skip to next predictor
                else:
                    # Equation 18
                    beta_j_hat = n / la.norm(X[:, pi], 2) ** 2
                    beta_j_hat = beta_j_hat * soft_thres(x_j_t_dot_r_minus_j_over_n, (1 - alpha) * lam)

                    cond_19 = la.norm(
                        soft_thres(compute_w_j(X, Z, pi).T @ (r_minus_j - X[:, pi] * beta_j_hat), alpha * lam),
                        2
                    )
                    if cond_19 <= (1 - alpha) * lam:
                        # Step 2.b: beta_j_hat != 0, theta_j_hat == 0
                        self.beta[pi] = beta_j_hat
                        continue
                    else:
                        # beta_j_hat != 0 and theta_j_hat != 0
                        # Generalised gradient procedure to find beta_j_hat and theta_j_hat
                        # TODO (1/7/2019) Test if this works
                        params = np.hstack((self.beta[pi], self.theta[pi, :]))
                        assert len(params) == k+1
                        params = opt.fmin(
                            j_partial, params, (pi, self.beta, self.theta, X, Z, y, alpha, lam)
                        )
                        params[abs(params) < 1e-12] = 0
                        self.beta[pi] = params[0]
                        self.theta[pi, :] = params[1:]

            self.history.append(
                j(self.beta_0, self.theta_0, self.beta, self.theta, X, Z, y, alpha, lam)
            )

    def predict(self, X, Z):
        if self.beta is None:
            raise NotFittedError

        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
            Z = Z.values

        return model(self.beta_0, self.theta_0, self.beta, self.theta, X, Z)
