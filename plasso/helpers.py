import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LinearRegression


eps = 1e-4


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


def model(beta_0, theta_0, beta, theta, x, z):
    n, p, k = x.shape[0], x.shape[1], z.shape[1]

    intercepts = beta_0 + (z @ theta_0)

    shared_model = x @ beta

    pliable = np.zeros(n)
    for j_i in range(p):
        w_j = compute_w_j(x, z, j_i)
        pliable = pliable + (w_j @ theta[j_i, :])

    return intercepts + shared_model + pliable


def partial_model(beta_0, theta_0, beta, theta, x, z, ignore_j):
    beta[ignore_j] = 0.0
    theta[ignore_j, :] = 0.0
    return model(beta_0, theta_0, beta, theta, x, z)


def j(beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
    n = len(y)

    mse = (1/(2*n)) * ((y - model(beta_0, theta_0, beta, theta, x, z))**2).sum()
    beta_matrix = v2a(beta)

    penalty_1 = la.norm(theta, 1).sum()
    penalty_2 = la.norm(np.hstack([beta_matrix, theta]), 2, axis=1).sum()
    penalty_3 = la.norm(theta, 2, axis=1).sum()

    cost = mse + (1 - alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3
    return cost


def j_partial(params_j, j_i, beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
    beta_j, theta_j = params_j[0], params_j[1:]
    beta[j_i] = beta_j
    theta[j_i, :] = theta_j
    return j(beta_0, theta_0, beta, theta, x, z, y, alpha, lam)


def derivative_wrt_beta_j(beta_0, theta_0, beta, theta, x, z, y, j, alpha, lam):
    y_hat = model(beta_0, theta_0, beta, theta, x, z)
    r = y - y_hat
    vec = np.hstack((beta[j], theta[j, :])) + eps
    u = beta[j] / la.norm(vec, 2)
    return (-1/len(r)) * x[:, j].T @ r + (1 - alpha) * lam * u


def derivative_wrt_theta_j(w_j, r, alpha, lam, u2, u3, v):
    # TODO (1/8/2019) compute this derivative to hopefully help optimisation
    return (-1/len(r)) * w_j.T @ r + (1 - alpha) * lam * (u2 + u3) + alpha * lam * v
