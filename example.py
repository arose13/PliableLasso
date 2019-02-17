import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from plasso import PliableLasso
from plasso.oldHelpers import PliableLassoModelHelper, lam_max
from plasso.PliableLasso import OPTIMISE_COORDINATE, OPTIMISE_CONVEX
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as graph


if __name__ == '__main__':
    # Setup
    n = 1000
    p = 50
    k = 4

    beta_0 = 0
    theta_0 = np.zeros(k)

    beta = np.zeros(p)
    beta[:4] = [2, -2, 2, 2]
    print(beta)

    theta = np.zeros((p, k))
    theta[2, 0] = 2.0
    theta[3, 1] = -2.0
    print(theta)

    z = stats.bernoulli(p=0.5).rvs(size=(n, k))
    print(z.shape)

    x = stats.norm().rvs(size=(n, p))
    print(x.shape)

    # Step 1. Check Lasso equivalent
    # Lasso Test
    y = x @ beta
    print(y.shape)

    func = PliableLassoModelHelper()
    y_hat = func.model(beta_0, theta_0, beta, np.zeros((p, k)), x, z)
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

    y_hat = func.model(beta_0, theta_0, beta, theta, x, z)
    assert_almost_equal(y_hat, y)
    print('Model correctly computes Pliables')
    print(f'R2 = {r2_score(y, y_hat):0.2%}, MSE = {mean_squared_error(y, y_hat):0.5f}')
    print()

    y_gt = y.copy()
    y += 0.5 * stats.norm().rvs(n)  # Add noise from paper

    lambda_max = lam_max(x, y, 0.5)
    lambda_min = 1e-3 * lambda_max
    print(f'\nlambda range [{lambda_min}, {lambda_max}]')

    # Optimisation Test (Convex Optimisation)
    plasso = PliableLasso(min_lam=0.5, fit_intercepts=False, verbose=False, max_iter=100)

    # print('\n=== Fitting Model via Convex Optimisation ===')
    # plasso.fit(x, z, y, optimizer=OPTIMISE_CONVEX)
    # y_hat = plasso.predict(x, z)
    #
    # print('\n== Outputs ==')
    #
    # print('\nbeta_0')
    # print(plasso.beta_0)
    #
    # print('\ntheta_0')
    # print(plasso.theta_0)
    #
    # print('\nbeta')
    # print(np.round(plasso.beta, 2))
    #
    # print('\ntheta')
    # print(np.round(plasso.theta, 2))
    #
    # print('--- Best Possible ---')
    # print(f'R2 = {r2_score(y, y_gt):0.2%}, MSE = {mean_squared_error(y, y_gt):0.5f}')
    # print(f'J = {objective(beta_0, theta_0, beta, theta, x, z, y, alpha=plasso.alpha, lam=plasso.lam):0.5f}')
    #
    # print('--- Obtained ---')
    # print(f'R2 = {r2_score(y, y_hat):0.2%}, MSE = {mean_squared_error(y, y_hat):0.5f}')
    # print(f'J = {plasso.cost(x, z, y):0.5f}')

    # Optimisation Test (Coordinate Descent)
    print('\n=== Fitting Model via Coordinate Descent ===')
    plasso.fit(x, z, y, optimizer=OPTIMISE_COORDINATE)
    y_hat = plasso.predict(x, z)

    graph.plot()

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
    print(f'J = {func.objective(beta_0, theta_0, beta, theta, x, z, y, alpha=plasso.alpha, lam=plasso.min_lam):0.5f}')

    print('--- Obtained ---')
    print(f'R2 = {r2_score(y, y_hat):0.2%}, MSE = {mean_squared_error(y, y_hat):0.5f}')
    print(f'J = {plasso.cost(x, z, y):0.5f}')
