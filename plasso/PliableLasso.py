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
    def __init__(self, alpha=0.5, n_lam=50, max_iter=500, min_lam=0, fit_intercepts=False, verbose=False):
        self.min_lam, self.alpha = min_lam, alpha
        self.n_lam = n_lam
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

    def fit(self, X, Z, y, optimizer=OPTIMISE_COORDINATE):
        if optimizer == OPTIMISE_CONVEX:
            return self._fit_convex_optimization(X, Z, y)
        elif optimizer == OPTIMISE_COORDINATE:
            return self._fit_coordinate_descent(X, Z, y)
        else:
            raise ValueError(f'Only allowed optimizers are {OPTIMISE_COORDINATE} or {OPTIMISE_CONVEX}')

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
        lam.value = self.min_lam

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
        self._reset_paths_dict()
        n, p = X.shape
        k = Z.shape[1]

        # Step 1: Setup
        alpha = self.alpha  # So I don't have to keep writing self
        beta_0, theta_0, beta, theta = 0.0, np.zeros(k), np.zeros(p), np.zeros((p, k))

        # Same lambda path specs as GLMnet (Apparently faster than solving for a single lambda
        lambda_max = lam_max(X, y, alpha)
        lambda_min = 1e-3 * lambda_max
        lambda_path = np.logspace(np.log10(lambda_max), np.log10(lambda_min), self.n_lam)
        lambda_path = lambda_path[lambda_path >= self.min_lam]
        self.paths['lam'] = lambda_path

        # Step 2: Update coefficients
        for lam in lambda_path:  # NOTE: This means you are ignoring the self.lam value
            # Update Paths
            self.paths['beta_0'].append(beta_0)
            self.paths['theta_0'].append(theta_0)
            self.paths['beta'].append(beta)
            self.paths['theta'].append(theta)

            # w_list = [compute_w_j(X, Z, j) for j in range(p)]

            for i in range(self.max_iter):
                # TODO 1/14/2019 estimate beta_0 and theta_0
                beta_new, theta_new = beta.copy(), theta.copy()
                iter_prev_score = objective(beta_0, theta_0, beta_new, theta_new, X, Z, y, alpha, lam)

                # Iterate through all p features
                tolerance = 1e-3
                for j in range(p):
                    x_j = X[:, j]
                    r_min_j = y - partial_model(beta_0, theta_0, beta, theta, X, Z, j)
                    w_j = compute_w_j(X, Z, j)  # w_list[j]

                    cond_17a = np.abs(x_j.T @ r_min_j / n) <= (1-alpha) * lam
                    cond_17b = la.norm(soft_thres(w_j.T @ r_min_j / n, alpha * lam), 2) <= 2 * (1-alpha) * lam

                    if cond_17a and cond_17b:
                        if self.verbose >= 4:
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
                                beta_j_hat = beta_new[j]  # Was this a problem?
                                theta_j_hat = theta_new[j, :]

                                grad_beta_j = -np.sum(x_j * r) / n
                                grad_theta_j = -w_j.T @ r / n
                                if self.verbose >= 3:
                                    print('--> Gradients')
                                    print(grad_beta_j)
                                    print(grad_theta_j)

                                beta_j_new_hat, theta_j_new_hat = solve_abg(
                                    beta_j_hat, theta_j_hat,
                                    grad_beta_j, grad_theta_j,
                                    alpha, lam, t
                                )

                                if self.verbose >= 3:
                                    print(f'new beta_{j} =', beta_j_new_hat)
                                    print(f'new theta_{j} =', theta_j_new_hat)

                                # Update Coefs
                                beta_new[j] = beta_j_new_hat
                                theta_new[j, :] = theta_j_new_hat
                                objective_current = objective(beta_0, theta_0, beta_new, theta_new, X, Z, y, alpha, lam)
                                improvement = objective_prev - objective_current

                                if abs(improvement) < tolerance:
                                    if self.verbose >= 2:
                                        print(f'==> Step converged j = {j} : {improvement}')
                                    break
                                else:
                                    objective_prev = objective_current

                            else:
                                warnings.warn(f'==> Max Steps reached (Failed to converged) on beta_{j}')

                    beta, theta = beta_new.copy(), theta_new.copy()

                if self.verbose >= 1:
                    from sklearn.metrics import r2_score
                    score_i = r2_score(y, model(beta_0, theta_0, beta, theta, X, Z))
                    print(f'==> Iter {i} @ lam {lam:0.3f} = {score_i:0.2%}')

                iter_current_score = objective(beta_0, theta_0, beta_new, theta_new, X, Z, y, alpha, lam)
                if abs(iter_prev_score - iter_current_score) < tolerance:
                    if self.verbose >= 1:
                        print(f'==> Converged on lam_i = {lam:0.3f}\n')
                    break
                else:
                    if self.verbose >= 1:
                        print(f'delta J_{i} = {iter_prev_score - iter_current_score}')
                    iter_prev_score = iter_current_score

        # Finally update model's parameters
        self.beta_0, self.theta_0, self.beta, self.theta = beta_0, theta_0, beta, theta
        for var in ['theta_0', 'beta', 'theta']:
            self.paths[var] = np.stack(self.paths[var])

    def _fallback_to_coordinate_descent(self, X, Z, y):
        warnings.warn('cvxpy is required for convex optimisation. Falling back to coordinate descent')
        return self._fit_coordinate_descent(X, Z, y)

    def _reset_paths_dict(self):
        self.paths = {}
        for var in ['lam', 'beta_0', 'theta_0', 'beta', 'theta']:
            self.paths[var] = []

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
        return objective(self.beta_0, self.theta_0, self.beta, self.theta, X, Z, y, self.alpha, self.min_lam)
