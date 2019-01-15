import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LinearRegression


def all_close_to(a, value=0, tol=1e-7):
    return np.all(abs(a - value) < tol)


def v2a(a):
    return a.reshape((len(a), 1))


def lam_max(x, y, alpha):
    n = len(y)
    dots = np.zeros(x.shape[1])
    for j in range(len(dots)):
        dots[j] = x[:, j].T @ y
    return np.abs(dots).max() / (n*alpha)


def soft_thres(x, thres):
    return np.sign(x) * np.maximum(np.abs(x) - thres, 0)


def quad_solution(u, v, w):
    temp = np.sqrt((v**2) - (4*u*w))
    root1 = (-v + temp) / (2*u)
    root2 = (-v - temp) / (2*u)
    return root1, root2


def solveabg(beta_j, theta_j, grad_beta, grad_theta, alpha, lam, tt):
    big, eps = 10e9, 1e-3

    g1 = beta_j - tt * grad_beta

    scrat = theta_j - tt * grad_theta

    tt2 = tt * alpha * lam

    scrat2 = soft_thres(scrat, tt2)

    ng1 = np.abs(g1)
    ng2 = np.sqrt(scrat2 @ scrat2)

    cc = tt * (1-alpha) * lam

    root1, root2 = quad_solution(1, 2*cc, 2*cc*ng2-ng1**2-ng2**2)

    a = np.array([
        ng1 * root1 / (cc + root1),
        ng1 * root2 / (cc + root2),
        ng1 * root1 / (cc + root2),
        ng1 * root2 / (cc + root1)
    ])
    b = np.array([
        root1 * (cc - ng2) / (cc + root1),
        root2 * (cc - ng2) / (cc + root2),
        root1 * (cc - ng2) / (cc + root2),
        root2 * (cc - ng2) / (cc + root1)
    ])

    xmin = big
    jhat, khat = 0, 0
    val1 = np.zeros((4, 4))
    val2 = val1.copy()
    for j in range(4):
        for k in range(4):
            val1[j, k] = big
            val2[j, k] = big

            den = np.sqrt(a[j]**2 + b[k]**2)
            if den > 0:
                val1[j, k] = (1+(cc/den)) * a[j] - ng1
                val2[j, k] = (1+cc*(1/b[k] + 1/den)) * b[k] - ng2
                temp = np.abs(val1[j, k]) + np.abs(val2[j, k])
                if temp < xmin:
                    jhat, khat = j, k
                    xmin = temp

    if np.abs(xmin) > eps:
        print('Failed to Solve ABG')

    if a[jhat] < 0 or b[khat] < 0:
        print('Failed: One of the norms are negative')

    xnorm = np.sqrt(a[jhat]**2 + b[khat]**2)

    beta_j_hat = (beta_j - tt * grad_beta) / (1 + cc/xnorm)

    scrat2 = theta_j - tt * grad_theta
    theta_j_hat = soft_thres(scrat2, tt2)
    theta_j_hat = theta_j_hat / (1 + cc * (1/xnorm + 1/b[khat]))

    return beta_j_hat, theta_j_hat


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


def objective(beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
    n = len(y)

    mse = (1/(2*n)) * la.norm(y - model(beta_0, theta_0, beta, theta, x, z), 2)**2
    beta_matrix = v2a(beta)

    penalty_1 = la.norm(np.hstack([beta_matrix, theta]), 2, axis=1).sum()
    penalty_2 = la.norm(theta, 2, axis=1).sum()
    penalty_3 = np.abs(theta).sum()  # sum(theta_jk)

    cost = mse + (1 - alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3
    return cost


def partial_objective(params_j, j_i, beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
    beta_j, theta_j = params_j[0], params_j[1:]
    beta[j_i] = beta_j
    theta[j_i, :] = theta_j
    return objective(beta_0, theta_0, beta, theta, x, z, y, alpha, lam)


def compute_r(beta_0, theta_0, beta, theta, x, z, y):
    return y - model(beta_0, theta_0, beta, theta, x, z)


def derivative_wrt_beta_j(beta_0, theta_0, beta, theta, x, z, y, j, alpha, lam):
    r = compute_r(beta_0, theta_0, beta, theta, x, z, y)
    beta_j_theta_j = np.hstack((beta[j], theta[j, :]))
    u = 0 if all_close_to(beta_j_theta_j, 0) else beta[j] / la.norm(beta_j_theta_j, 2)
    return (-1/len(r)) * x[:, j].T @ r + (1 - alpha) * lam * u


def derivative_wrt_theta_j(beta_0, theta_0, beta, theta, x, z, y, j, alpha, lam):
    beta_j = beta[j]
    theta_j = theta[j, :]

    r = compute_r(beta_0, theta_0, beta, theta, x, z, y)
    w_j = compute_w_j(x, z, j)

    beta_j_theta_j = np.hstack((beta_j, theta_j))
    u2 = 0 if all_close_to(beta_j_theta_j, 0) else theta_j / la.norm(beta_j_theta_j, 2)

    u3 = 0 if all_close_to(theta_j, 0) else theta_j / la.norm(theta_j, 2)

    v = np.sign(theta[j, :])

    return (-1/len(r)) * w_j.T @ r + (1 - alpha) * lam * (u2 + u3) + alpha * lam * v
