import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LinearRegression


def v2a(a):
    return a.reshape((len(a), 1))


def soft_thres(x, thres):
    return np.sign(x) * np.clip(abs(x) - thres, a_min=0, a_max=None)


def quad_solution(u, v, w):
    temp = np.sqrt((v**2) - (4*u*w))
    root1 = (-v + temp) / (2*u)
    root2 = (-v - temp) / (2*u)
    return root1, root2


def compute_w_j(x, z, j: int):
    # TODO 12/23/2018 add caching decorating for when the function inputs are the same
    k = z.shape[1]
    x_j = np.tile(v2a(x[:, j]), (1, k))
    return x_j * z


def estimate_intercepts(z, y):
    lm = LinearRegression()
    lm.fit(z, y)

    beta_0 = lm.intercept_
    theta_0 = lm.coef_
    y = y - lm.predict(z)
    return beta_0, theta_0, y


def model(beta_0, theta_0, beta, theta, x, z, ignore_j=None):
    n, p, k = x.shape[0], x.shape[1], z.shape[1]
    ignore_intercepts = beta_0 is np.nan or theta_0 is np.nan

    intercepts = 0 if ignore_intercepts else beta_0 + (z @ theta_0)

    if ignore_j is not None:
        beta[ignore_j] = 0
    shared_model = x @ beta

    pliable = np.zeros(n)
    for j_i in range(p):  # TODO 12/23/2018 this for loops does not need to be here if you dot everything
        if j_i == ignore_j:
            continue
        w_j = compute_w_j(x, z, j_i)
        pliable = pliable + (w_j @ theta[j_i, :])

    return intercepts + shared_model + pliable


def partial_model(beta, theta, x, z, ignore_j):
    return model(np.nan, np.nan, beta, theta, x, z, ignore_j)


def j(beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
    n = len(y)

    mse = (0.5 * n) * (y - model(beta_0, theta_0, beta, theta, x, z)).sum()
    beta_matrix = v2a(beta)

    penalty_1 = la.norm(theta, 1).sum()
    penalty_2 = la.norm(np.hstack([beta_matrix, theta]), 2, axis=1).sum()
    penalty_3 = la.norm(theta, 2, axis=1).sum()

    return mse + (1 - alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3


def j_partial(params_j, j_i, beta, theta, x, z, y, alpha, lam):
    beta_j, theta_j = params_j[0], params_j[1:]
    beta[j_i] = beta_j
    theta[j_i, :] = theta_j
    return j(np.nan, np.nan, beta, theta, x, z, y, alpha, lam)


def derivative_with_respect_to_beta_j(x_j, r, alpha, lam, u):
    return (-1/len(r)) * x_j.T @ r + (1 - alpha) * lam * u


def derivative_with_respect_to_theta_j(w_j, r, alpha, lam, u2, u3, v):
    return (-1/len(r)) * w_j.T @ r + (1 - alpha) * lam * (u2 + u3) + alpha * lam * v
