import numpy as np
import pandas as pd
import numpy.linalg as la
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

    def fit(self, X, Z, y):
        # NOTE: wtf this is an O(nk)
        alpha, lam = self.alpha, self.lam  # So I don't have to keep writing self
        n, p = X.shape
        k = Z.shape[1]

        # Step 1: Compute beta_0 and theta_0
        self.beta, self.theta = np.zeros(p), np.zeros((p, k))
        self.beta_0, self.theta_0, y = estimate_intercepts(Z, y)

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
                        beta_j, theta_j = self.beta[pi], self.theta[pi, :]
                        l2_beta_j_theta_j = la.norm(np.hstack([beta_j, theta_j]), 2)
                        w_j = compute_w_j(X, Z, pi)

                        r = y - model(self.beta_0, self.theta_0, self.beta, self.theta, X, Z)
                        u = beta_j / l2_beta_j_theta_j
                        u2 = theta_j / l2_beta_j_theta_j
                        u3 = theta_j / la.norm(theta_j, 2)
                        v = np.sign(theta_j)
                        t = 1.0  # Just trying this as a number

                        grad_beta_j = (-1/n) * v2a(X[:, pi]).T @ r + (1-alpha) * lam * u
                        grad_theta_j = (-1/n) * w_j.T @ r + (1-alpha) * lam * (u2 + u3) + alpha * lam * v

                        c = t * (1 - alpha) * lam
                        g1 = abs(-t * grad_beta_j * l)
                        g2 = la.norm(soft_thres(-t * grad_theta_j * l, t * alpha * lam), 2)
                        u_prime = np.nan
                        u_double_prime = np.nan
                        v_prime = np.nan

                        a_hat = (g1 * u_prime) / (c + u_double_prime)
                        b_hat = (g1 * v_prime * (c - g2)) / (c + v_prime)
                        l2_gamma_j = np.sqrt((a_hat ** 2) + (b_hat ** 2))

                        c1 = 1 + c / l2_gamma_j
                        c2 = 1 + c * ((1/b_hat) + (1 / l2_gamma_j))

                        # TODO 12/26/2018 update coefs
                        self.beta[pi] = (-t * grad_beta_j * l) / c1
                        self.theta[pi, :] = la.norm(soft_thres(-t * grad_theta_j * l, t * alpha * lam), 2) / c2
                        # TODO 12/26/2018 do I loop this update logic until it doesn't need to run again?

    def predict(self, X, Z):
        if self.beta is None:
            raise NotFittedError

        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
            Z = Z.values

        return model(self.beta_0, self.theta_0, self.beta, self.theta, X, Z)
