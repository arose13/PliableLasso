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


def solve_abg(b, grad_beta, grad_theta, alpha, lam, tt=0.1):
    """
    Solve the system of equations for update in the pliable lasso.
    This is step 4b

    :param b: Single beta with length of modifying vars k (from theta) TODO is this a correct interpretation?
    :param grad_beta: Real(1) -> gradient beta = -sum(xk*r)/n  where xk is X.
            where xk is X column of in current coord descent step, r=current residual
    :param grad_theta: Real(k) -> gradient theta = -t(X_k Z) * r/n
    :param alpha: Real(1) [0, 1] mixing parameter
    :param lam: Real(1)+ -> regularisation strength
    :param tt: Real(1) -> backtracking parameter
    :return:
    """
    big, eps = 1e10, 1e-3
    k = len(grad_theta)

    g1 = -tt * grad_beta

    scratch = np.zeros(k)
    for i in range(k):  # TODO (1/2/2019) vectorize?
        scratch[i] = b[i] - tt * grad_theta[i]

    tt2 = tt * alpha * lam
    scratch = soft_thres(scratch, tt2)

    ng1 = np.abs(g1)
    ng2 = np.sqrt(scratch @ scratch)

    cc = tt * (1-alpha) * lam

    root1, root2 = quad_solution(u=1.0, v=2 * cc, w=2 * cc * ng2 - ng1**2 - ng2**2)

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
    k = z.shape[1]

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


def derivative_with_respect_to_beta_j(x_j, r, alpha, lam, u):
    return (-1/len(r)) * x_j.T @ r + (1 - alpha) * lam * u


def derivative_with_respect_to_theta_j(w_j, r, alpha, lam, u2, u3, v):
    return (-1/len(r)) * w_j.T @ r + (1 - alpha) * lam * (u2 + u3) + alpha * lam * v
