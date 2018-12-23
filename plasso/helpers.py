import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LinearRegression


def v2a(a):
    return a.reshape((len(a), 1))


def soft_thres(x, thres):
    return np.sign(x) * np.clip(abs(x) - thres, a_min=0, a_max=None)


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


def model(beta_0, theta_0, beta, theta, x, z):
    n, p, k = x.shape[0], x.shape[1], z.shape[1]

    intercepts = beta_0 + (z @ theta_0)

    shared_model = x @ beta

    pliable = np.zeros(n)
    for j_i in range(p):  # TODO 12/23/2018 this for loops does not need to be here if you dot everything
        w_j = compute_w_j(x, z, j_i)
        pliable = pliable + (w_j @ theta[j_i, :])

    return intercepts + shared_model + pliable


def partial_model(beta, theta, x, z, partial_index):
    n, p = x.shape

    beta = beta.copy()
    beta[partial_index] = 0

    pliable = np.zeros(n)
    for j_i in range(p):
        if j_i == partial_index:
            continue
        else:
            x_j = np.tile(v2a(x[:, j_i]), (1, k))
            pliable = pliable + ((x_j * z) @ theta[j_i, :])

    return x @ beta + pliable


def j(beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
    n = len(y)

    mse = (0.5 * n) * (y - model(beta_0, theta_0, beta, theta, x, z)).sum()
    beta_matrix = v2a(beta)

    penalty_1 = la.norm(theta, 1).sum()
    penalty_2 = la.norm(np.hstack([beta_matrix, theta]), 2, axis=1).sum()
    penalty_3 = la.norm(theta, 2, axis=1).sum()

    return mse + (1 - alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3