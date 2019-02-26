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
    def __init__(self, alpha=0.5, eps=1e-2, n_lam=50, min_lam=0,
                 max_interaction_terms=500, max_iter=100, fit_intercepts=True,
                 verbose=False):
        self.min_lam, self.alpha, self.eps = min_lam, alpha, eps
        self.n_lam = n_lam
        self.max_iter, self.max_interaction_terms = max_iter, max_interaction_terms
        self.fit_intercepts = fit_intercepts

        # Model coefficients
        self.beta_0 = None
        self.theta_0 = None
        self.beta = None
        self.theta = None

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

        # Step 1: Initial Setup
        n, p = X.shape
        k = Z.shape[1]
        beta_0, theta_0, beta, theta = 0.0, np.zeros(k), np.zeros(p), np.zeros((p, k))

        # Solve lambda path spec
        lambda_max, lambda_min = lam_min_max(X, y, self.alpha, 1e-2)
        lambda_path = np.logspace(np.log10(lambda_max), np.log10(lambda_min), self.n_lam)
        lambda_path = lambda_path[lambda_path >= self.min_lam]
        if len(lambda_path) == 0:
            raise ValueError(f'`min_lam` was set too high! Maximum lambda for this problem is {lambda_max}')
        self.min_lam = max(lambda_path.min(), self.min_lam)

        # Step 2: Update coefficients with coordinate descent
        result = coordinate_descent(
            X, Z, y,
            beta_0, theta_0, beta, theta,
            self.alpha, lambda_path,
            self.max_iter, self.max_interaction_terms, self.fit_intercepts,
            self.verbose
        )

        # Step 3: Save results
        var_names = ['lam', 'beta_0', 'theta_0', 'beta', 'theta']
        self.paths = {var_name: var_list for var_name, var_list in zip(var_names, result)}
        for var in var_names[2:]:
            self.paths[var] = np.stack(self.paths[var])

        # Step 4: Select best coefficients
        # TODO (2/21/2019) replace with WAY better logic
        self.beta_0 = self.paths['beta_0'][-1]
        self.theta_0 = self.paths['theta_0'][-1]
        self.beta = self.paths['beta'][-1]
        self.theta = self.paths['theta'][-1]

        return self

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

        return model(self.beta_0, self.theta_0, self.beta, self.theta, X, Z, precomputed_w=compute_w(X, Z))

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
