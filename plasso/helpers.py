import numpy as np
import numpy.linalg as la
from numba import njit
from functools import partial
# TODO (2/21/2019) - to allow prediction after the model is trained
# TODO (2/21/2019) - Automatically terminate training if interaction condition is met


def placebo():
    def wrapper(func):
        return func
    return wrapper


njit = partial(njit, cache=True)
# njit = placebo


def lam_min_max(x, y, alpha, eps=1e-2):
    """
    Approximate the minimum and maximum values for the lambda

    :param x:
    :param y:
    :param alpha:
    :param eps:
    :return:
    """
    assert 0 < eps < 1, '`eps` must be between 0 and 1'
    n, p = x.shape
    dots = np.zeros(p)
    for j in range(p):
        dots[j] = x[:, j].T @ y
    lam_max = np.abs(dots).max() / (n*alpha)
    lam_min = eps * lam_max
    return lam_max, lam_min


@njit()
def concat_beta_theta(beta, theta):
    matrix = np.zeros((beta.shape[0], theta.shape[1]))
    matrix[:, :-1] = theta
    matrix[:, -1] = beta


@njit()
def soft_thres(x, thres):
    return np.sign(x) * np.maximum(np.abs(x) - thres, 0)


@njit()
def quad_solution(u, v, w):
    temp = ((v**2) - (4*u*w))**0.5
    root1 = (-v + temp) / (2*u)
    root2 = (-v - temp) / (2*u)
    return root1, root2


@njit()
def solve_abg(beta_j, theta_j, grad_beta, grad_theta, alpha, lam, t):
    """
    Solves a and b so gradient iterations of theta are not needed. Equation (22)

    Numba compile from 2500ms to 70ms

    :param beta_j:
    :param theta_j:
    :param grad_beta:
    :param grad_theta:
    :param alpha:
    :param lam:
    :param t: Backtracking parameter. Controlled in the fit loop currently set to 0.1
    :return:
    """
    # Convergence hyperparameters
    big, eps = 10e9, 1e-3

    g1 = np.abs(beta_j - t * grad_beta)

    g2_thres = t * alpha * lam
    c = t * (1 - alpha) * lam

    scrat = soft_thres(theta_j - t * grad_theta, g2_thres)
    g2 = np.sqrt(scrat @ scrat)  # Confirmed that this is a faster l2 norm than la.norm(x, 2)

    root1, root2 = quad_solution(1, 2 * c, 2 * c * g2 - g1 ** 2 - g2 ** 2)

    a = np.array([
        g1 * root1 / (c + root1),
        g1 * root2 / (c + root2),
        g1 * root1 / (c + root2),
        g1 * root2 / (c + root1)
    ])
    b = np.array([
        root1 * (c - g2) / (c + root1),
        root2 * (c - g2) / (c + root2),
        root1 * (c - g2) / (c + root2),
        root2 * (c - g2) / (c + root1)
    ])

    x_min = big
    j_hat, k_hat = 0, 0
    for j in range(4):
        for k in range(4):
            denominator = (a[j] ** 2 + b[k] ** 2) ** 0.5  # l2 norm
            if denominator > 0:
                val1 = (1 + (c / denominator)) * a[j] - g1
                val2 = (1 + c * (1 / b[k] + 1 / denominator)) * b[k] - g2

                temp = abs(val1) + abs(val2)  # l1 norm
                if temp < x_min:
                    j_hat, k_hat = j, k
                    x_min = temp

    xnorm = np.sqrt(a[j_hat] ** 2 + b[k_hat] ** 2)

    beta_j_hat = (beta_j - t * grad_beta) / (1 + c / xnorm)

    scrat = theta_j - t * grad_theta
    theta_j_hat = soft_thres(scrat, g2_thres)
    theta_j_hat = theta_j_hat / (1 + c * (1 / xnorm + 1 / b[k_hat]))

    return beta_j_hat, theta_j_hat


@njit()
def compute_w_j(x, z, j):
    w_j = np.zeros(z.shape)
    for k_i in range(z.shape[1]):
        w_j[:, k_i] = x[:, j] * z[:, k_i]
    return w_j


@njit()
def compute_w(x, z):
    return [compute_w_j(x, z, j) for j in range(x.shape[1])]


@njit()
def compute_pliable(x, theta, precomputed_w):
    n, p = x.shape
    pliable = np.zeros(n)
    for j in range(p):
        if np.any(theta[j, :]):
            w_j = precomputed_w[j]
            pliable += w_j @ theta[j, :]
    return pliable


@njit()
def model(beta_0, theta_0, beta, theta, x, z, precomputed_w):
    """
    The pliable lasso model described in the paper
    y ~ f(x)

    formulated as

    y ~ b_0 + Z theta_0 + X b + \sum( w_j theta_ji )

    :param precomputed_w: List of all precomputed Wj
    :param beta_0:
    :param theta_0:
    :param beta:
    :param theta:
    :param x:
    :param z:
    :return:
    """
    intercepts = beta_0 + (z @ theta_0)
    shared_model = x @ beta
    pliable = compute_pliable(x, theta, precomputed_w)
    return intercepts + shared_model + pliable


@njit()
def model_min_j(beta_0, theta_0, beta, theta, x, z, j, precomputed_w):
    """
    y ~ f(x) with X_j removed from the model

    :param precomputed_w:
    :param beta_0:
    :param theta_0:
    :param beta:
    :param theta:
    :param x:
    :param z:
    :param j: The jth predictor to ignore.
    :return:
    """
    beta[j] = 0.0
    theta[j, :] = 0.0
    return model(beta_0, theta_0, beta, theta, x, z, precomputed_w)


@njit()
def model_j(beta_j, theta_j, x, precomputed_w, j):
    """
    r_j ~ beta_j * X_j + W_j @ theta_j

    Only a the residual fit on a single predictor is made
    """
    return beta_j * x[:, j] + precomputed_w[j] @ theta_j


@njit()
def objective(beta_0, theta_0, beta, theta, x, z, y, alpha, lam, precomputed_w):
    """
    Full objective function J(beta, theta) described in the paper

    :param precomputed_w: List of all precomputed Wj
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
    n, p = x.shape
    k = z.shape[1]

    mse = (1 / (2 * n)) * la.norm(y - model(beta_0, theta_0, beta, theta, x, z, precomputed_w), 2) ** 2

    # TODO (2/19/2019) replace with concat_beta_theta() function
    coef_matrix = np.zeros((p, k+1))
    coef_matrix[:, :-1] = theta
    coef_matrix[:, -1] = beta

    # Numba does not support the axis kwarg on la.norm() [axis 1 means operates across rows]
    penalty_1, penalty_2 = 0.0, 0.0
    for j in range(p):
        penalty_1 += la.norm(coef_matrix[j, :], 2)
        penalty_2 += la.norm(theta[j, :], 2)
    penalty_3 = np.abs(theta).sum()

    cost = mse + (1-alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3
    return cost


@njit()
def penalties_min_j(beta_0, theta_0, beta, theta, x, z, y, precomputed_w, ignore_j):
    # Compute the MSE, penalty 1 and penalty 2 when the jth predictor is not in the model.
    n, p = x.shape
    k = z.shape[1]

    mse = (1 / (2*n)) * la.norm(y - model_min_j(beta_0, theta_0, beta, theta, x, z, ignore_j, precomputed_w), 2) ** 2

    coef_matrix = np.zeros((p, k + 1))
    coef_matrix[:, :-1] = theta
    coef_matrix[:, -1] = beta

    # Ignore the jth modifier from the model
    coef_matrix[ignore_j, :] = 0.0
    theta[ignore_j, :] = 0.0

    # Compute penalties
    penalty_1, penalty_2 = 0.0, 0.0
    for j in range(p):
        penalty_1 += la.norm(coef_matrix[j, :], 2)
        penalty_2 += la.norm(theta[j, :], 2)
    penalty_3 = np.abs(theta).sum()

    return mse, penalty_1, penalty_2, penalty_3


@njit()
def partial_objective(beta_j, theta_j, x, r_min_j, precomputed_w, j, alpha, lam, mse, penalty_1, penalty_2, penalty_3):
    # This only computes the objective for the jth modifier variables
    n, p = x.shape
    k = theta_j.shape[0]

    # Compute only the residual fit of the model since everything else is the same
    r_hat = model_j(beta_j, theta_j, x, precomputed_w, j)
    mse += (1 / (2*n)) * la.norm(r_min_j - r_hat, 2)**2

    # Penalty 1
    coef_vector = np.zeros(k+1)
    coef_vector[:-1] = theta_j
    coef_vector[-1] = beta_j
    penalty_1 += la.norm(coef_vector, 2)

    # Penalty 2
    penalty_2 += la.norm(theta_j, 2)

    # Penalty 3
    penalty_3 += np.abs(theta_j).sum()

    cost = mse + (1-alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3
    return cost


@njit()
def coordinate_descent(x, z, y, beta_0, theta_0, beta, theta, alpha, lam_path, max_iter, max_interaction_terms):
    n, p = x.shape
    precomputed_w = compute_w(x, z)

    # Lists
    lam_list = []
    beta_0_list = []
    theta_0_list = []
    beta_list = []
    theta_list = []

    tolerance = 1e-6
    for nth_lam, lam in enumerate(lam_path):
        for i in range(max_iter):
            iter_prev_score = objective(beta_0, theta_0, beta, theta, x, z, y, alpha, lam, precomputed_w)

            # Iterate through all p features
            for j in range(p):
                x_j = x[:, j]
                r_min_j = y - model_min_j(beta_0, theta_0, beta, theta, x, z, j, precomputed_w)
                w_j = precomputed_w[j]

                # Check if beta_j == 0 and theta_j == 0
                cond_17a = np.abs(x_j.T @ r_min_j / n) <= (1-alpha) * lam
                cond_17b = la.norm(soft_thres(w_j.T @ r_min_j / n, alpha * lam), 2) <= 2 * (1-alpha) * lam

                if cond_17a and cond_17b:
                    # beta_j == 0 and theta_j == 0
                    pass
                else:
                    beta_j_hat = (n / la.norm(x_j, 2)**2) * soft_thres(x_j.T @ r_min_j / n, (1-alpha) * lam)

                    cond_19 = la.norm(soft_thres(w_j.T @ (r_min_j - x_j * beta_j_hat) / n, alpha * lam), 2)
                    cond_19 = cond_19 <= (1-alpha) * lam

                    if cond_19:
                        # beta_j != 0 and theta_j == 0
                        beta[j] = beta_j_hat
                    else:
                        # beta_j != 0 and theta_j != 0
                        t, l, eps = 0.1, 1.0, 1e-5
                        precomputed_penalties_minus_j = penalties_min_j(
                            beta_0, theta_0, beta, theta, x, z, y, precomputed_w, j
                        )
                        pc_mse, pc_penalty_1, pc_penalty_2, pc_penalty_3 = precomputed_penalties_minus_j
                        objective_prev = partial_objective(
                            beta[j], theta[j, :],
                            x, r_min_j, precomputed_w, j,
                            alpha, lam,
                            pc_mse, pc_penalty_1, pc_penalty_2, pc_penalty_3
                        )
                        for _ in range(100):  # Max steps
                            beta_j_hat = beta[j]
                            theta_j_hat = theta[j, :]
                            r = r_min_j - model_j(beta_j_hat, theta_j_hat, x, precomputed_w, j)

                            grad_beta_j = -np.sum(x_j * r) / n
                            grad_theta_j = -w_j.T @ r / n

                            beta_j_hat, theta_j_hat = solve_abg(
                                beta_j_hat, theta_j_hat,
                                grad_beta_j, grad_theta_j,
                                alpha, lam, t
                            )

                            # Update coefficients
                            beta[j] = beta_j_hat
                            theta[j, :] = theta_j_hat

                            objective_current = partial_objective(
                                beta[j], theta[j],
                                x, r_min_j, precomputed_w, j,
                                alpha, lam,
                                pc_mse, pc_penalty_1, pc_penalty_2, pc_penalty_3
                            )
                            improvement = objective_prev - objective_current
                            if abs(improvement) < tolerance:
                                # Converged
                                break
                            else:
                                objective_prev = objective_current

            iter_current_score = objective(beta_0, theta_0, beta, theta, x, z, y, alpha, lam, precomputed_w)
            if abs(iter_prev_score - iter_current_score) < tolerance:
                break  # Converged on lam_i

        # Check maximum interaction terms reached. If so early stop just like Tibs.
        # if theta[theta != 0].size > max_interaction_terms:
        #     break

        # Save coefficients
        lam_list.append(lam)
        beta_0_list.append(beta_0)
        theta_0_list.append(theta_0.copy())
        beta_list.append(beta.copy())
        theta_list.append(theta.copy())

    # Return results
    return lam_list, beta_0_list, theta_0_list, beta_list, theta_list
