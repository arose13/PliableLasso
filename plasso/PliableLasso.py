import warnings
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .helpers import *


OPTIMISE_CONVEX = 'convex'
OPTIMISE_COORDINATE = 'coordinate'


# noinspection PyPep8Naming
class PliableLasso(BaseEstimator):
    """
    Pliable Lasso https://arxiv.org/pdf/1712.00484.pdf
    """
    def __init__(self, lam=1.0, alpha=0.5, max_iter=500, fit_intercepts=False, verbose=False):
        self.lam, self.alpha = lam, alpha
        self.max_iter = max_iter
        self.fit_intercepts = fit_intercepts

        # Model coefs
        self.beta_0 = None
        self.theta_0 = None
        self.beta = None
        self.theta = None

        # Metrics
        self.verbose = verbose
        self.history = []
        self.paths = {}

    def fit(self, X, Z, y, optimizer=OPTIMISE_CONVEX):
        if optimizer == OPTIMISE_CONVEX:
            return self._fit_convex_optimization(X, Z, y)
        else:
            return self._fit_coordinate_descent(X, Z, y)

    def _fallback_to_coordinate_descent(self, X, Z, y):
        warnings.warn('cvxpy is required for convex optimisation. Falling back to coordinate descent')
        return self._fit_coordinate_descent(X, Z, y)

    def _fit_convex_optimization(self, X, Z, y):
        try:
            import cvxpy as cvx
            from .cvxHelpers import objective_cvx
        except ModuleNotFoundError:
            return self._fallback_to_coordinate_descent(X, Z, y)

        # Check CVXPY version number
        from distutils.version import LooseVersion
        if LooseVersion(cvx.__version__) < LooseVersion('1.0.11'):
            return self._fallback_to_coordinate_descent(X, Z, y)

        self.history = []

        # Hyperparameters
        alpha = self.alpha
        lam = cvx.Parameter(nonneg=True)
        lam.value = self.lam

        # Parameters
        n, p = X.shape
        k = Z.shape[1]
        if self.fit_intercepts:
            beta_0 = cvx.Variable(1)
            theta_0 = cvx.Variable(k)
        else:
            beta_0 = 0.0
            theta_0 = np.zeros(k)
        beta = cvx.Variable(p)
        theta = cvx.Variable((p, k))

        # Fit with Convex Optimisation
        problem = cvx.Problem(
            cvx.Minimize(objective_cvx(beta_0, theta_0, beta, theta, X, y, Z, alpha, lam))
        )
        # Solve on a decreasing lambda path
        problem.solve(verbose=self.verbose, solver=cvx.CVXOPT, max_iter=self.max_iter)

        self.beta_0 = beta_0 if self.fit_intercepts else beta_0.value
        self.theta_0 = theta_0 if self.fit_intercepts else theta_0.value
        self.beta = beta.value
        self.theta = theta.value

        return self

    def _fit_coordinate_descent(self, X, Z, y):
        n, p = X.shape
        k = Z.shape[1]

        # Step 1: Setup
        alpha, lam = self.alpha, self.lam  # So I don't have to keep writing self
        beta_0, theta_0, beta, theta = 0.0, np.zeros(k), np.zeros(p), np.zeros((p, k))

        # Step 2: Update coefficients
        for i in range(self.max_iter):
            # TODO 1/14/2019 estimate beta_0 and theta_0
            beta_new, theta_new = beta.copy(), theta.copy()
            iter_prev_score = objective(beta_0, theta_0, beta_new, theta_new, X, Z, y, alpha, lam)

            # Iterate through all p features
            tolerance = 1e-3
            for j in range(p):
                x_j = X[:, j]
                r_min_j = y - partial_model(beta_0, theta_0, beta, theta, X, Z, j)
                w_j = compute_w_j(X, Z, j)

                cond_17a = np.abs(x_j.T @ r_min_j / n) <= (1-alpha) * lam
                cond_17b = la.norm(soft_thres(w_j.T @ r_min_j / n, alpha * lam), 2) <= 2 * (1-alpha) * lam

                if cond_17a and cond_17b:
                    if self.verbose >= 2:
                        print(f'beta_{j} == 0 and theta_{j} == 0')

                else:
                    beta_j_hat = (n / la.norm(x_j, 2)**2) * soft_thres(x_j.T @ r_min_j / n, (1-alpha) * lam)

                    cond_19 = la.norm(soft_thres(w_j.T @ (r_min_j - x_j * beta_j_hat) / n, alpha * lam), 2)
                    cond_19 = cond_19 <= (1-alpha) * lam

                    if cond_19:
                        beta_new[j] = beta_j_hat
                        if self.verbose >= 2:
                            print(f'beta_{j} != 0 and theta_{j} == 0')
                            print(f'-> {beta_j_hat}')

                    else:
                        if self.verbose >= 2:
                            print(f'beta_{j} != 0 and theta_{j} != 0')
                        t, l, eps = 0.1, 1.0, 1e-5
                        max_steps = 100
                        objective_prev = objective(beta_0, theta_0, beta_new, theta_new, X, Z, y, alpha, lam)
                        for _ in range(max_steps):
                            r = y - model(beta_0, theta_0, beta_new, theta_new, X, Z)
                            theta_j_hat = theta_new[j, :]
                            l2_gamma_hat = np.sqrt(beta_j_hat**2 + la.norm(theta_j_hat, 2)**2)

                            grad_beta_j = -np.sum(x_j * r) / n
                            grad_theta_j = -w_j.T @ r / n
                            if self.verbose >= 3:
                                print('--> Gradients')
                                print(grad_beta_j)
                                print(grad_theta_j)

                            beta_j_new_hat = beta_0 - t * grad_beta_j * l
                            beta_j_new_hat /= (1 + (t * (1-alpha) * lam) / l2_gamma_hat)

                            theta_j_new_hat = soft_thres(theta_0 - t * grad_theta_j * l, t * alpha * lam)
                            theta_j_new_hat /= (1 + t*(1-alpha)*lam*((1/la.norm(theta_j_hat, 2)) + (1/l2_gamma_hat)))
                            if self.verbose >= 3:
                                print('New beta =', beta_j_new_hat)
                                print('New theta =', theta_j_new_hat)

                            # Update Coefs
                            beta_new[j] = beta_j_new_hat
                            theta_new[j, :] = theta_j_new_hat
                            objective_current = objective(beta_0, theta_0, beta_new, theta_new, X, Z, y, alpha, lam)
                            improvement = objective_prev - objective_current

                            if abs(improvement) < tolerance:
                                if self.verbose >= 2:
                                    print(f'==> Converged {improvement}')
                                break
                            else:
                                objective_prev = objective_current

                        else:
                            warnings.warn(f'==> Max Steps reached (Failed to converged) on beta_{j}')

            beta, theta = beta_new.copy(), theta_new.copy()

            if self.verbose >= 1:
                from sklearn.metrics import r2_score
                print(f'==> Iter {i} = {r2_score(y, model(beta_0, theta_0, beta, theta, X, Z)):0.2%}')

            #
            iter_current_score = objective(beta_0, theta_0, beta_new, theta_new, X, Z, y, alpha, lam)
            if abs(iter_prev_score - iter_current_score) < tolerance:
                if self.verbose >= 1:
                    print('Early Termination')
                break
            else:
                if self.verbose >= 1:
                    print(f'J_{i} = {iter_prev_score - iter_current_score}')
                iter_prev_score = iter_current_score

        # Finally update model's parameters
        self.beta_0, self.theta_0, self.beta, self.theta = beta_0, theta_0, beta, theta

    def predict(self, X, Z):
        if self.beta is None:
            raise NotFittedError

        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
            Z = Z.values

        return model(self.beta_0, self.theta_0, self.beta, self.theta, X, Z)

    def score(self, X, Z, y):
        from sklearn.metrics import r2_score
        return r2_score(y, model(self.beta_0, self.theta_0, self.beta, self.theta, X, Z))

    def cost(self, X, Z, y):
        return objective(self.beta_0, self.theta_0, self.beta, self.theta, X, Z, y, self.alpha, self.lam)
