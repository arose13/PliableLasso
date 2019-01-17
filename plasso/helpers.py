import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LinearRegression


def all_close_to(a, value=0, tol=1e-7):
    return np.all(abs(a - value) < tol)


def v2a(a):
    return a.reshape((len(a), 1))


def lam_max(x, y, alpha):
    """
    Approximate the lowest value for lambda where all the coefficients will be zero

    :param x:
    :param y:
    :param alpha:
    :return:
    """
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


def solve_abg(beta_j, theta_j, grad_beta, grad_theta, alpha, lam, tt):
    """
    Solves a and b so gradient iterations of theta are not needed.

    :param beta_j:
    :param theta_j:
    :param grad_beta:
    :param grad_theta:
    :param alpha:
    :param lam:
    :param tt:
    :return:
    """
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
    for j in range(4):
        for k in range(4):
            den = np.sqrt(a[j]**2 + b[k]**2)
            if den > 0:
                val1 = (1 + (cc / den)) * a[j] - ng1
                val2 = (1 + cc * (1 / b[k] + 1 / den)) * b[k] - ng2

                temp = np.abs(val1) + np.abs(val2)
                if temp < xmin:
                    jhat, khat = j, k
                    xmin = temp

    if np.abs(xmin) > eps:
        print('Failed to Solve ABG')

    if a[jhat] < 0 or b[khat] < 0:
        print('Warning! One of the norms are negative [Solve ABG]')

    xnorm = np.sqrt(a[jhat]**2 + b[khat]**2)

    beta_j_hat = (beta_j - tt * grad_beta) / (1 + cc/xnorm)

    scrat2 = theta_j - tt * grad_theta
    theta_j_hat = soft_thres(scrat2, tt2)
    theta_j_hat = theta_j_hat / (1 + cc * (1/xnorm + 1/b[khat]))

    return beta_j_hat, theta_j_hat


def estimate_intercepts(z, y):
    lm = LinearRegression()
    lm.fit(z, y)

    beta_0 = lm.intercept_
    theta_0 = lm.coef_
    y = y - lm.predict(z)
    return beta_0, theta_0, y


# noinspection PyPep8Naming
class PliableLassoModelHelper:
    """
    This class allows for runtime optimised coordinate descent by trying to more aggressively cache results
    """
    def __init__(self, X=None, Z=None):
        self.w_j_list = None  # Dont use an if else statement since compute_w_j() needs this to exist first.
        if X is not None and Z is not None:
            self.w_j_list = [self.compute_w_j(X, Z, j) for j in range(X.shape[1])]

    def model(self, beta_0, theta_0, beta, theta, x, z):
        """
        The pliable lasso model described in the paper
        y ~ f(x)

        formulated as

        y ~ b_0 + Z theta_0 + X b + \sum( w_j theta_ji )

        :param beta_0:
        :param theta_0:
        :param beta:
        :param theta:
        :param x:
        :param z:
        :return:
        """
        n, p, k = x.shape[0], x.shape[1], z.shape[1]

        intercepts = beta_0 + (z @ theta_0)

        shared_model = x @ beta

        pliable = np.zeros(n)
        # For performance, check if there are even nonzero values in theta
        if np.any(theta):
            # At least 1 nonzero value in theta
            for j_i in range(p):
                # For performance, screen if theta_j is nonzero before computing pliable
                if np.any(theta[j_i, :]):
                    w_j = self.compute_w_j(x, z, j_i)
                    pliable = pliable + (w_j @ theta[j_i, :])

        return intercepts + shared_model + pliable

    def partial_model(self, beta_0, theta_0, beta, theta, x, z, ignore_j):
        """
        y ~ f(x) with X_j removed from the model

        :param beta_0:
        :param theta_0:
        :param beta:
        :param theta:
        :param x:
        :param z:
        :param ignore_j:
        :return:
        """
        beta[ignore_j] = 0.0
        theta[ignore_j, :] = 0.0
        return self.model(beta_0, theta_0, beta, theta, x, z)

    def objective(self, beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
        """
        Full objective function J(beta, theta) described in the paper

        :param beta_0:
        :param theta_0:
        :param beta:
        :param theta:
        :param x:
        :param z:
        :param y:
        :param alpha:
        :param lam:
        :return:
        """
        n = len(y)

        mse = (1 / (2 * n)) * la.norm(y - self.model(beta_0, theta_0, beta, theta, x, z), 2) ** 2
        beta_matrix = v2a(beta)

        penalty_1 = la.norm(np.hstack([beta_matrix, theta]), 2, axis=1).sum()
        penalty_2 = la.norm(theta, 2, axis=1).sum()
        penalty_3 = np.abs(theta).sum()

        cost = mse + (1 - alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3
        return cost

    def partial_objective(self, params_j, j_i, beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
        """
        Computes the cost function J with feature _j removed from the model

        :param params_j:
        :param j_i:
        :param beta_0:
        :param theta_0:
        :param beta:
        :param theta:
        :param x:
        :param z:
        :param y:
        :param alpha:
        :param lam:
        :return:
        """
        beta_j, theta_j = params_j[0], params_j[1:]
        beta[j_i] = beta_j
        theta[j_i, :] = theta_j
        return self.objective(beta_0, theta_0, beta, theta, x, z, y, alpha, lam)

    def compute_w_j(self, x, z, j: int):
        """
        Performs the elementwise multiplication of X_j with Z

        :param x:
        :param z:
        :param j:
        :return:
        """
        if self.w_j_list is None:
            w_j = np.zeros(z.shape)
            for k_i in range(z.shape[1]):
                w_j[:, k_i] = x[:, j] * z[:, k_i]
            return w_j
        else:
            return self.w_j_list[j]

    def derivative_wrt_beta_j(self, beta_0, theta_0, beta, theta, x, z, y, j, alpha, lam):
        """
        Full Derivative with respect to beta_j as described in the paper

        :param beta_0:
        :param theta_0:
        :param beta:
        :param theta:
        :param x:
        :param z:
        :param y:
        :param j:
        :param alpha:
        :param lam:
        :return:
        """
        r = y - self.model(beta_0, theta_0, beta, theta, x, z)
        beta_j_theta_j = np.hstack((beta[j], theta[j, :]))
        u = 0 if all_close_to(beta_j_theta_j, 0) else beta[j] / la.norm(beta_j_theta_j, 2)
        return (-1 / len(r)) * x[:, j].T @ r + (1 - alpha) * lam * u

    def derivative_wrt_theta_j(self, beta_0, theta_0, beta, theta, x, z, y, j, alpha, lam):
        """
        Full Derivative with respect to theta_j as described in the paper

        :param beta_0:
        :param theta_0:
        :param beta:
        :param theta:
        :param x:
        :param z:
        :param y:
        :param j:
        :param alpha:
        :param lam:
        :return:
        """
        beta_j = beta[j]
        theta_j = theta[j, :]

        r = y - self.model(beta_0, theta_0, beta, theta, x, z)
        w_j = self.compute_w_j(x, z, j)

        beta_j_theta_j = np.hstack((beta_j, theta_j))
        u2 = 0 if all_close_to(beta_j_theta_j, 0) else theta_j / la.norm(beta_j_theta_j, 2)

        u3 = 0 if all_close_to(theta_j, 0) else theta_j / la.norm(theta_j, 2)

        v = np.sign(theta[j, :])

        return (-1 / len(r)) * w_j.T @ r + (1 - alpha) * lam * (u2 + u3) + alpha * lam * v
