import numpy as np
from time import time
from numpy.testing import assert_almost_equal
from scipy import stats
from plasso import PliableLasso
from plasso.numbaSolver import model, objective, compute_w
from plasso.PliableLasso import lam_min_max
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as graph


if __name__ == '__main__':
    # Setup
    np.random.seed(2010)
    n = 1000
    p = 50
    k = 4

    beta_0 = 0.0
    theta_0 = np.zeros(k).astype(float)

    beta = np.zeros(p)
    beta[:4] = [2, -2, 2, 2]
    print(beta)

    theta = np.zeros((p, k))
    theta[2, 0] = 2.0
    theta[3, 1] = -2.0
    print(theta)

    z = stats.bernoulli(p=0.5).rvs(size=(n, k)).astype(float)
    print(z.shape)

    x = stats.norm().rvs(size=(n, p))
    print(x.shape)

    # Step 1. Check Lasso equivalent
    # Lasso Test
    y = x @ beta
    print(y.shape)

    y_hat = model(beta_0, theta_0, beta, np.zeros((p, k)).astype(float), x, z, compute_w(x, z))
    assert_almost_equal(y_hat, y)
    print('Model does reduce to Lasso')
    print(f'R2 = {r2_score(y, y_hat):0.2%}, MSE = {mean_squared_error(y, y_hat):0.5f}')
    print()

    # Pliable Lasso Test
    y = x[:, 0] * beta[0]
    y += x[:, 1] * beta[1]
    y += x[:, 2] * (beta[2] + 2*z[:, 0])
    y += x[:, 3] * (beta[3] - 2*z[:, 1])
    print(y.shape)

    y_hat = model(beta_0, theta_0, beta, theta, x, z, compute_w(x, z))
    assert_almost_equal(y_hat, y)
    print('Model correctly computes Pliables')
    print(f'R2 = {r2_score(y, y_hat):0.2%}, MSE = {mean_squared_error(y, y_hat):0.5f}')
    print()

    y_gt = y.copy()
    y += 0.5 * stats.norm().rvs(n)  # Add noise from paper

    lambda_max, lambda_min = lam_min_max(x, z, y, 0.5)
    print(f'\nlambda range [{lambda_min}, {lambda_max}]')

    # Optimisation Test (Coordinate Descent)
    plasso = PliableLasso(cv=0.1, verbose=True)

    print('\n=== Fitting Model via Coordinate Descent ===')
    start_time = time()
    plasso.fit(x, z, y)
    end_time = time()
    print(f'Fit time = {end_time - start_time:.5f}s')

    y_hat = plasso.predict(x, z)

    plasso.plot_coef_paths()
    graph.show()

    plasso.plot_interaction_paths()
    graph.show()

    plasso.plot_intercepts_path()
    graph.show()

    plasso.plot_score_path()
    graph.show()

    print('\n== Outputs ==')

    print('\nbeta_0')
    print(plasso.beta_0)

    print('\ntheta_0')
    print(plasso.theta_0)

    print('\nbeta')
    print(plasso.beta.round(2))

    print('\ntheta')
    print(plasso.theta.round(2))

    print('--- Best Possible ---')
    print(f'R2 = {r2_score(y, y_gt):0.2%}, MSE = {mean_squared_error(y, y_gt):0.5f}')
    print(f'J = {objective(beta_0, theta_0, beta, theta, x, z, y, plasso.alpha, plasso.min_lam, compute_w(x, z)):0.5f}')

    print('--- Obtained ---')
    print(f'R2 = {r2_score(y, y_hat):0.2%}, MSE = {mean_squared_error(y, y_hat):0.5f}')
    print(f'J = {plasso.cost(x, z, y):0.5f}')
