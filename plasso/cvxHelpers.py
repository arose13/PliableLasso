import numpy as np
import cvxpy as cvx
from .helpers import v2a


def compute_w_j(x, z, j):
    k = z.shape[1]
    x_j = np.tile(v2a(x[:, j]), (1, k))
    return x_j * z


def y_hat(beta_0, theta_0, beta, theta, x, z):
    # Confirmed DCP and Convex
    n, p = x.shape

    intercepts = beta_0 + (z @ theta_0)

    shared_model = x @ beta

    pliables = np.zeros(n)
    for j_i in range(p):
        w_j = compute_w_j(x, z, j_i)
        pliables = pliables + (w_j @ theta[j_i, :])

    output = intercepts + shared_model + pliables
    return output


def objective_cvx(beta_0, theta_0, beta, theta, x, y, z, alpha, lam):
    n, p = x.shape

    # mse = (1/(2*n)) * cvx.sum((y - y_hat(beta_0, theta_0, beta, theta, x, z)) ** 2)
    mse = cvx.norm((y - y_hat(beta_0, theta_0, beta, theta, x, z)), p=2)**2

    beta_matrix = cvx.reshape(beta, (p, 1))

    penalty_1 = cvx.sum(cvx.norm(cvx.hstack([beta_matrix, theta]), p=2, axis=1))
    penalty_2 = cvx.sum(cvx.norm(theta, p=2, axis=1))
    penalty_3 = cvx.sum(cvx.norm(theta, p=1))

    loss = mse + (1 - alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3
    return loss