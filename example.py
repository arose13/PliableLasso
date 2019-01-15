import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from plasso import PliableLasso
from plasso.helpers import model, objective, derivative_wrt_beta_j, derivative_wrt_theta_j
from plasso.PliableLasso import OPTIMISE_COORDINATE
from sklearn.metrics import r2_score, mean_squared_error


if __name__ == '__main__':
    # Setup
    n = 1000
    p = 50
    k = 4

    beta_0 = 0
    theta_0 = np.zeros(k)

    beta = np.zeros(p)
    beta[:4] = [2, -2, 2, 2]

    theta = np.zeros((p, k))
    theta[2, 0] = 2.0
    theta[3, 1] = -2.0

    z = stats.bernoulli(p=0.5).rvs(size=(n, k))
    print(z.shape)

    x = stats.norm().rvs(size=(n, p))
    print(x.shape)

    # Step 1. Check Lasso equivalent
    # Lasso Test
    y = x @ beta
    print(y.shape)

    y_hat = model(beta_0, theta_0, beta, np.zeros((p, k)), x, z)
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

    y_hat = model(beta_0, theta_0, beta, theta, x, z)
    assert_almost_equal(y_hat, y)
    print('Model correctly computes Pliables')
    print(f'R2 = {r2_score(y, y_hat):0.2%}, MSE = {mean_squared_error(y, y_hat):0.5f}')
    print()

    # Derivative Test
    print('=== Analytical Derivative ===')
    for pi in range(p):
        d_wrt_bj = derivative_wrt_beta_j(0, np.zeros(k), np.zeros(p), np.zeros((p, k)), x, z, y, pi, 0.5, 2)
        print(f'dJ/dB_{pi} = {d_wrt_bj}')

        d_wrt_tj = derivative_wrt_theta_j(0, np.zeros(k), np.zeros(p), np.zeros((p, k)), x, z, y, pi, 0.5, 2)
        print(f'dJ/dT_{pi} = {d_wrt_tj}')
        break

    # Optimisation Test
    y_gt = y.copy()
    y += 0.5 * stats.norm().rvs(n)  # Add noise from paper

    print('\n=== Fitting Model ===')
    plasso = PliableLasso(lam=0.001, fit_intercepts=False, max_iter=100)
    plasso.fit(x, z, y, optimizer=OPTIMISE_COORDINATE)
    y_hat = plasso.predict(x, z)

    print('\n=== Outputs ===')

    print('\nbeta_0')
    print(plasso.beta_0)

    print('\ntheta_0')
    print(plasso.theta_0)

    print('\nbeta')
    print(np.round(plasso.beta, 2))

    print('\ntheta')
    print(np.round(plasso.theta, 2))

    print('--- Best Possible ---')
    print(f'R2 = {r2_score(y, y_gt):0.2%}, MSE = {mean_squared_error(y, y_gt):0.5f}')
    print(f'J = {objective(beta_0, theta_0, beta, theta, x, z, y, alpha=plasso.alpha, lam=plasso.lam):0.5f}')

    print('--- Obtained ---')
    print(f'R2 = {r2_score(y, y_hat):0.2%}, MSE = {mean_squared_error(y, y_hat):0.5f}')
    print(f'J = {plasso.cost(x, z, y):0.5f}')
