import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_X_y
from sklearn.preprocessing.data import _handle_zeros_in_scale

from .helpers import *

from sklearn.linear_model import LinearRegression, Lasso

OPTIMISE_CONVEX = 'convex'
OPTIMISE_COORDINATE = 'coordinate'


def lam_min_max(x, y, alpha, eps=1e-2, cv=1):
    """
    Approximate the minimum and maximum values for the lambda

    :param x:
    :param y:
    :param alpha:
    :param cv: Number used to estimate the actual sample size the solver will see.
    :param eps:
    :return:
    """
    assert 0 < eps < 1, '`eps` must be between 0 and 1'

    if cv > 1 and isinstance(cv, int):
        scale = (cv-1)/cv
    elif 0 < cv < 1:
        scale = 1-cv
    else:
        scale = 1.0

    n, p = x.shape
    dots = np.zeros(p)
    for j in range(p):
        dots[j] = x[:, j].T @ (y - y.mean())
    lam_max = np.abs(dots).max() / (n*scale*alpha)
    lam_min = eps * lam_max
    return lam_max, lam_min


def _binary_columns(a: np.ndarray):
    """
    Returns a vector of where binary columns are in matrix a
    """
    assert a.ndim == 2, 'Input is not a matrix, {}'.format(a)
    is_binary = [len(np.unique(a[:, j])) == 2 for j in range(a.shape[1])]
    return np.array(is_binary)


def _preprocess_x_z_y(x, z, y, normalize):
    # Check for binary traits
    x_binary_columns, z_binary_columns = _binary_columns(x), _binary_columns(z)

    # Offsetting
    if normalize:
        # Swap with np.average if I want sample weights
        x_offset, z_offset, y_offset = [a.mean(axis=0) for a in (x, z, y)]
        x_offset[x_binary_columns] = 0.0
        z_offset[z_binary_columns] = 0.0

        x -= x_offset
        z -= z_offset
        y -= y_offset
    else:
        x_offset = np.zeros(x.shape[1])
        z_offset = np.zeros(z.shape[1])
        y_offset = 0.0

    # Scaling
    if normalize:
        x_scale, z_scale, y_scale = [a.std(axis=0) for a in (x, z, y)]
        x_scale, z_scale, y_scale = [_handle_zeros_in_scale(a) for a in (x_scale, z_scale, y_scale)]
        x_scale[x_binary_columns] = 1.0
        z_scale[z_binary_columns] = 1.0

        x /= x_scale
        z /= z_scale
    else:
        x_scale = np.ones(x.shape[1], dtype=x.dtype)
        z_scale = np.ones(z.shape[1], dtype=z.dtype)
        y_scale = 1.0

    return x, z, y, x_offset, z_offset, y_offset, x_scale, z_scale, y_scale


def _transform_solved_model_parameters(coordinate_descent_results, x_mu, x_sd, z_mu, z_sd, y_mu, y_sd):
    # Some Setup: Compute the pooled standard deviation
    p, k = len(x_sd), len(z_sd)
    sd_xx = np.tile(x_sd, (k, 1)).T
    sd_zz = np.tile(z_sd, (p, 1))
    sd_xz = sd_xx * sd_zz

    beta_0_updated, theta_0_updated, beta_updated, theta_updated = [[] for _ in range(4)]
    for _, beta_0, theta_0, beta, theta in zip(*coordinate_descent_results):
        theta = theta / sd_xz

        beta = (beta / x_sd) - (theta @ z_mu)

        theta_0 = (theta_0 / z_sd) - (x_mu @ theta)

        beta_0 = y_mu + beta_0 - (theta @ z_mu @ x_mu) - (z_mu @ theta_0) - (x_mu @ beta)

        # Create new lists
        beta_updated.append(beta)
        beta_0_updated.append(beta_0)
        theta_updated.append(theta)
        theta_0_updated.append(theta_0)

    # Updated coordinate descent results
    return coordinate_descent_results[0], beta_0_updated, theta_0_updated, beta_updated, theta_updated


# noinspection PyPep8Naming
class PliableLasso(BaseEstimator):
    """
    Pliable Lasso https://arxiv.org/pdf/1712.00484.pdf
    """
    def __init__(self, alpha=0.5, eps=1e-2, n_lam=50, min_lam=0,
                 max_interaction_terms=500, max_iter=100, cv=3, metric=mean_squared_error, normalize=True,
                 verbose=False):
        self.min_lam, self.alpha, self.eps = min_lam, alpha, eps
        self.n_lam = n_lam
        self.max_iter, self.max_interaction_terms = max_iter, max_interaction_terms
        self.cv, self.metric = cv, metric
        self.normalize = normalize

        # Model coefficients
        self.beta_0 = None
        self.theta_0 = None
        self.beta = None
        self.theta = None

        # Scaling terms
        self.x_sd, self.x_mu = [None] * 2
        self.z_sd, self.z_mu = [None] * 2
        self.y_mu = None

        # Metrics
        self.verbose = verbose
        self.paths = {}

    def fit(self, X, Z, y, optimizer=OPTIMISE_COORDINATE):
        # Convert all types to floats before proceeding
        X, Z, y = X.astype(float), Z.astype(float), y.astype(float)

        if optimizer == OPTIMISE_CONVEX:
            return self._fit_convex_optimization(X, Z, y)
        elif optimizer == OPTIMISE_COORDINATE:
            return self._fit_coordinate_descent(X, Z, y)
        else:
            raise ValueError('Only allowed optimizers are {} or {}'.format(OPTIMISE_COORDINATE, OPTIMISE_CONVEX))

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

        # Hyperparameters
        alpha = self.alpha
        lam = cvx.Parameter(nonneg=True)
        lam.value = self.min_lam

        # Parameters
        n, p = X.shape
        k = Z.shape[1]

        beta_0 = cvx.Variable(1)
        theta_0 = cvx.Variable(k)
        beta = cvx.Variable(p)
        theta = cvx.Variable((p, k))

        # Fit with Convex Optimisation
        problem = cvx.Problem(
            cvx.Minimize(objective_cvx(beta_0, theta_0, beta, theta, X, y, Z, alpha, lam))
        )
        # Solve on a decreasing lambda path
        problem.solve(verbose=self.verbose, solver=cvx.CVXOPT, max_iter=self.max_iter)

        self.beta_0 = beta_0.value
        self.theta_0 = theta_0.value
        self.beta = beta.value
        self.theta = theta.value

        return self

    def _fit_coordinate_descent(self, X, Z, y):
        # Step 0: Input checking
        X, y = check_X_y(X, y, dtype=[np.float64, np.float32], y_numeric=True)
        Z, _ = check_X_y(Z, y, dtype=[np.float64, np.float32])
        self._reset_paths_dict_and_variables()

        # Step 1: Initial Setup
        Xt, Zt, yt, x_mu, z_mu, y_mu, x_sd, z_sd, y_sd = _preprocess_x_z_y(
            X.copy(), Z.copy(), y.copy(),
            self.normalize
        )
        self.x_sd, self.x_mu = x_sd, x_mu
        self.z_sd, self.z_mu = z_sd, z_mu
        self.y_sd, self.y_mu = y_sd, y_mu

        n, p = X.shape
        k = Z.shape[1]
        beta_0, theta_0, beta, theta = 0.0, np.zeros(k), np.zeros(p), np.zeros((p, k))

        # Solve lambda path spec
        lambda_max, lambda_min = lam_min_max(X, y, self.alpha, self.eps, self.cv)
        lambda_path = np.logspace(np.log10(lambda_max), np.log10(lambda_min), self.n_lam)
        lambda_path = lambda_path[lambda_path >= self.min_lam]
        if len(lambda_path) == 0:
            raise ValueError('`min_lam` was set too high! Maximum lambda for this problem is {}'.format(lambda_max))
        self.min_lam = max(lambda_path.min(), self.min_lam)

        # Step 2: Update coefficients with coordinate descent
        if self.cv > 1 and isinstance(self.cv, int):
            # Cross Validation
            k_fold_cv = KFold(self.cv)
            all_score_paths = []
            for indices_train, indices_test in k_fold_cv.split(X):
                result = coordinate_descent(
                    Xt[indices_train, :],
                    Zt[indices_train, :],
                    yt[indices_train],
                    0.0, np.zeros(k), np.zeros(p), np.zeros((p, k)),  # beta_0, theta_0, beta, theta,
                    self.alpha, lambda_path,
                    self.max_iter, self.max_interaction_terms,
                    self.verbose
                )

                result = _transform_solved_model_parameters(
                    result,
                    x_mu, x_sd,
                    z_mu, z_sd,
                    y_mu, y_sd
                )

                # Score all models on the lambda path
                all_score_paths.append(
                    self._score_models_on_lambda_path(
                        result,
                        X[indices_test, :], Z[indices_test, :], y[indices_test]
                    )
                )
            # Select best lambda
            all_score_paths = np.vstack(all_score_paths)
            score_path = all_score_paths.mean(axis=0)
            best_lambda_index = score_path.argmin()

        elif 0 < self.cv < 1:
            # Train Test Split
            indices_train, indices_test = train_test_split(np.arange(len(y)), test_size=self.cv)
            result = coordinate_descent(
                Xt[indices_train, :],
                Zt[indices_train, :],
                yt[indices_train],
                0.0, np.zeros(k), np.zeros(p), np.zeros((p, k)),  # beta_0, theta_0, beta, theta
                self.alpha, lambda_path,
                self.max_iter, self.max_interaction_terms,
                self.verbose
            )

            result = _transform_solved_model_parameters(
                result,
                x_mu, x_sd,
                z_mu, z_sd,
                y_mu, y_sd
            )

            # Select best lambda
            score_path = self._score_models_on_lambda_path(
                result,
                X[indices_test, :], Z[indices_test, :], y[indices_test]
            )
            best_lambda_index = score_path.argmin()

        else:
            # Simply solve the lambda path
            result = coordinate_descent(
                X, Z, y,
                beta_0, theta_0, beta, theta,
                self.alpha, lambda_path,
                self.max_iter, self.max_interaction_terms,
                self.verbose
            )
            score_path = np.array([0])
            best_lambda_index = -1

        # Step 3: Save the results
        # NOTE: result_i is only the results of the last CV pass. This is probably not the way to do it.
        var_names = ['lam', 'beta_0', 'theta_0', 'beta', 'theta']
        self.paths = {var_name: var_list for var_name, var_list in zip(var_names, result)}
        for var in var_names[2:]:
            self.paths[var] = np.stack(self.paths[var])
        self.paths['score'] = score_path

        # Step 4: Select best coefficients
        self.beta_0 = self.paths['beta_0'][best_lambda_index]
        self.theta_0 = self.paths['theta_0'][best_lambda_index]
        self.beta = self.paths['beta'][best_lambda_index]
        self.theta = self.paths['theta'][best_lambda_index]

        return self

    def _fallback_to_coordinate_descent(self, X, Z, y):
        warnings.warn('cvxpy is required for convex optimisation. Falling back to coordinate descent')
        return self._fit_coordinate_descent(X, Z, y)

    def _reset_paths_dict_and_variables(self):
        self.paths = {}
        for var in ['score', 'lam', 'beta_0', 'theta_0', 'beta', 'theta']:
            self.paths[var] = []
        self.beta_0, self.theta_0, self.beta, self.theta = [None] * 4

    def _score_models_on_lambda_path(self, coordinate_descent_results, x, z, y):
        # Variables come in this order ['lam', 'beta_0', 'theta_0', 'beta', 'theta']
        precomputed_w = compute_w(x, z)
        scores = []
        for lam_i, beta_0, theta_0, beta, theta in zip(*coordinate_descent_results):
            scores.append(self.metric(
                y_true=y,
                y_pred=model(beta_0, theta_0, beta, theta, x, z, precomputed_w)
            ))
        print(f'Last beta_0 => {beta_0}')
        return np.array(scores)

    def predict(self, X, Z, lam=None):
        X, Z = X.astype(float), Z.astype(float)

        if self.beta is None:
            raise NotFittedError

        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
            Z = Z.values

        # Actually make predictions
        if lam is None:
            # If None then use the most predictive lambda
            return model(
                self.beta_0,
                self.theta_0,
                self.beta,
                self.theta,
                X, Z, precomputed_w=compute_w(X, Z)
            )
        else:
            # This will select the lam on the lambda path closest to the selected lambda
            lam_index = find_nearest(self.paths['lam'], lam, return_idx=True)
            return model(
                self.paths['beta_0'][lam_index],
                self.paths['theta_0'][lam_index],
                self.paths['beta'][lam_index],
                self.paths['theta'][lam_index],
                X, Z, precomputed_w=compute_w(X, Z)
            )

    def score(self, X, Z, y):
        from sklearn.metrics import r2_score
        return r2_score(y, model(self.beta_0, self.theta_0, self.beta, self.theta, X, Z, compute_w(X, Z)))

    def cost(self, X, Z, y):
        return objective(
            self.beta_0, self.theta_0, self.beta, self.theta, X, Z, y, self.alpha, self.min_lam, compute_w(X, Z)
        )

    def plot_coef_paths(self):
        import matplotlib.pyplot as graph

        graph.plot(self.paths['lam'], self.paths['beta'], linewidth=1)
        graph.ylabel(r'$\beta$')
        graph.xlabel(r'$\lambda$')
        graph.xscale('log')

    def plot_intercepts_path(self):
        import matplotlib.pyplot as graph

        graph.plot(self.paths['lam'], self.paths['beta_0'], color='black', linewidth=2, label=r'$\beta_0$')
        graph.plot(self.paths['lam'], self.paths['theta_0'], label=r'$\theta_0$')
        graph.ylabel('Intercepts')
        graph.xlabel(r'$\lambda$')
        graph.xscale('log')

    def plot_interaction_paths(self):
        import matplotlib.pyplot as graph
        p = self.paths.get('beta', None)
        if p is None:
            raise NotFittedError
        p = p.shape[1]

        for j in range(p):
            if np.any(self.paths['theta'][:, j, :]):
                graph.plot(self.paths['lam'], self.paths['theta'][:, j, :], linewidth=1)

        graph.ylabel(r'$\theta$')
        graph.xlabel(r'$\lambda$')
        graph.xscale('log')

    def plot_score_path(self):
        import matplotlib.pyplot as graph

        graph.plot(self.paths['lam'], self.paths['score'])
        graph.axvline(
            self.paths['lam'][np.argmin(self.paths['score'])],
            linestyle='--', color='black', alpha=0.5, label='Best MSE'
        )
        graph.ylabel('MSE')
        graph.xlabel(r'$\lambda$')
        graph.xscale('log')
