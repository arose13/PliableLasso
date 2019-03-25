import numpy as np
import torch as pt


# PyTorch functions
def soft_thres(x, thres):
    return pt.sign(x) * pt.clamp(pt.abs(x) - thres, min=0)


def quad_solution(u, v, w):
    temp = ((v ** 2) - (4 * u * w)) ** 0.5
    root1 = (-v + temp) / (2 * u)
    root2 = (-v - temp) / (2 * u)
    return root1, root2


def solve_abg(beta_j, theta_j, grad_beta, grad_theta, alpha, lam, t):
    big = pt.ones(1) * 1e9
    eps = 1e-5

    g1 = pt.abs(beta_j - t * grad_beta)

    g2_thres = t * alpha * lam
    c = t * (1 - alpha) * lam

    scrat = soft_thres(theta_j - t * grad_theta, g2_thres)
    g2 = pt.norm(scrat, 2)

    root1, root2 = quad_solution(pt.ones(1), 2 * c, 2 * c * g2 - g1 ** 2 - g2 ** 2)

    a = [
        g1 * root1 / (c + root1),
        g1 * root2 / (c + root2),
        g1 * root1 / (c + root2),
        g1 * root2 / (c + root1)
    ]
    b = [
        root1 * (c - g2) / (c + root1),
        root2 * (c - g2) / (c + root2),
        root1 * (c - g2) / (c + root2),
        root2 * (c - g2) / (c + root1)
    ]

    # x_min, j_hat, k_hat = accelerate_abg_search(big, a, b, c, g1, g2)
    x_min = big
    j_hat, k_hat = 0, 0
    for j in range(4):
        for k in range(4):
            denominator = (a[j] ** 2 + b[k] ** 2) ** 0.5  # l2 norm
            if bool(denominator > 0):
                val1 = (1 + (c / denominator)) * a[j] - g1
                val2 = (1 + c * (1 / b[k] + 1 / denominator)) * b[k] - g2

                temp = pt.abs(val1) + pt.abs(val2)  # l1 norm
                if bool(temp < x_min):
                    j_hat, k_hat = j, k
                    x_min = temp

    # Check convergence
    is_converged = bool(pt.abs(x_min) < eps) or bool(a[j_hat] < 0) or bool(b[k_hat] < 0)

    xnorm = (a[j_hat] ** 2 + b[k_hat] ** 2) ** 0.5  # l2 norm

    beta_j_hat = (beta_j - t * grad_beta) / (1 + c / xnorm)

    scrat = theta_j - t * grad_theta
    theta_j_hat = soft_thres(scrat, g2_thres)
    theta_j_hat = theta_j_hat / (1 + c * ((1 / xnorm) + (1 / pt.abs(b[k_hat]))))  # Ensure b_hat norm is always positive

    return beta_j_hat, theta_j_hat, is_converged


def concat_beta_theta(beta, theta):
    p, k = theta.shape
    matrix = pt.zeros(p, k + 1)
    matrix[:, :-1] = theta
    matrix[:, -1] = beta
    return matrix


def compute_wj(xj, z):
    wj = pt.zeros_like(z)
    for ki in range(z.shape[1]):
        wj[:, ki] = xj * z[:, ki]
    return wj


def compute_pliable(x, z, theta):
    n, p = x.shape
    pliable = pt.zeros(n)
    for j in range(p):
        if pt.nonzero(theta[j, :]).numel() > 0:
            wj = compute_wj(x[:, j], z)
            pliable += wj @ theta[j, :]
    return pliable


def model(beta_0, theta_0, beta, theta, x, z):
    intercepts = beta_0 + (z @ theta_0)
    shared_model = x @ beta
    pliable = compute_pliable(x, z, theta)
    return intercepts + shared_model + pliable


def model_min_j(beta_0, theta_0, beta, theta, x, z, j):
    beta[j] = 0.0
    theta[j, :] = 0.0
    return model(beta_0, theta_0, beta, theta, x, z)


def model_j(beta_j, theta_j, x_j, w_j):
    return beta_j * x_j + w_j @ theta_j


def objective(beta_0, theta_0, beta, theta, x, z, y, alpha, lam):
    n, p = x.shape

    y_hat = model(beta_0, theta_0, beta, theta, x, z)
    mse = (1 / (2 * n)) * pt.norm(y - y_hat, 2) ** 2

    coef_matrix = concat_beta_theta(beta, theta)
    penalty_1 = pt.norm(coef_matrix, 2, dim=1).sum()
    penalty_2 = pt.norm(theta, 2, dim=1).sum()
    penalty_3 = pt.abs(theta).sum()

    return mse + (1 - alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3


def penalties_min_j(beta_0, theta_0, beta, theta, x, z, y, ignore_j):
    n, p = x.shape

    y_hat_min_j = model_min_j(beta_0, theta_0, beta, theta, x, z, ignore_j)
    mse = (1 / (2 * n)) * pt.norm(y - y_hat_min_j, 2) ** 2

    coef_matrix = concat_beta_theta(beta, theta)

    # Ignore jth modifier from the model
    coef_matrix[ignore_j, :] = 0.0
    theta[ignore_j, :] = 0.0

    # Compute penalties
    penalty_1 = pt.norm(coef_matrix, 2, dim=1).sum()
    penalty_2 = pt.norm(theta, 2, dim=1).sum()
    penalty_3 = pt.abs(theta).sum()

    return mse, penalty_1, penalty_2, penalty_3


def partial_objective(
        beta_j, theta_j, xj, z, r_min_j, wj,
        alpha, lam, mse, penalty_1, penalty_2, penalty_3
):
    n = xj.shape[0]
    k = z.shape[1]

    # Compute only the residual fit of the model since everything else is the same
    r_hat = model_j(beta_j, theta_j, xj, wj)
    mse += (1 / (2 * n)) * pt.norm(r_min_j - r_hat, 2)**2

    # Penalty 1
    coef_vector = pt.zeros(k + 1)
    coef_vector[:-1] = theta_j
    coef_vector[-1] = beta_j
    penalty_1 += pt.norm(coef_vector, 2)

    # Penalty 2
    penalty_2 += pt.norm(theta_j, 2)

    # Penalty 3
    penalty_3 += pt.abs(theta_j).sum()

    return mse + (1-alpha) * lam * (penalty_1 + penalty_2) + alpha * lam * penalty_3


def coordinate_descent_pytorch(x, z, y, alpha, lam_path, max_iter, max_interaction_terms, verbose):
    # Setup
    x = pt.from_numpy(x.astype(np.float32))
    z = pt.from_numpy(z.astype(np.float32))
    y = pt.from_numpy(y.astype(np.float32))
    alpha = pt.tensor(alpha)

    n, p = x.shape
    k = z.shape[1]

    # Initialize Variables
    beta_0 = pt.tensor(0.0)
    theta_0 = pt.zeros(k)
    beta = pt.zeros(p)
    theta = pt.zeros(p, k)

    lam_path = pt.from_numpy(lam_path.astype(np.float32))
    lam_max = lam_path.max()

    # Precomputed Variables
    w = pt.ones(n, k + 1)
    w[:, :-1] = z
    inv_w_w = pt.inverse(w.t() @ w + 1e-9 * pt.eye(k + 1))

    # Solve ABG Parameter
    t = 0.1 / (x ** 2).mean()
    if verbose:
        print('tt =', t)

    # Solution List
    lam_list = []
    beta_0_list = []
    theta_0_list = []
    beta_list = []
    theta_list = []

    # PyTorch Coordinate Descent
    tolerance = 1e-5
    for nth_lam, lam in enumerate(lam_path):
        for i in range(max_iter):
            iter_prev_score = objective(
                beta_0, theta_0, beta, theta,
                x, z, y,
                alpha, lam
            )

            # Compute beta_0 and theta_0 from the least square regression of the current residual on Z
            # Analytic solution has no lower sample bound (Z.T @ Z + cI)^-1 @ (Z.T @ r)
            r_current = y - model(0.0, pt.zeros(k), beta, theta, x, z)
            b = inv_w_w @ (w.t() @ r_current)
            theta_0 = b[:-1]
            beta_0 = b[-1]

            # Iterate across all p features
            r = y - model(beta_0, theta_0, beta, theta, x, z)
            for j in range(p):
                # SAFE screening rule (https://statweb.stanford.edu/~tibs/ftp/strong.pdf)
                if pt.abs(x[:, j] @ y) < 2 * lam - lam_max:
                    continue

                w_j = compute_wj(x[:, j], z)
                r_min_j = r + model_j(beta[j], theta[j, :], x[:, j], w_j)

                # Check if beta_j == 0 and theta_j == 0
                cond_17a = pt.abs(x[:, j] @ r_min_j / n) <= (1 - alpha) * lam
                cond_17b = pt.norm(soft_thres(w_j.t() @ r_min_j / n, alpha * lam), 2) <= 2 * (1 - alpha) * lam

                if cond_17a and cond_17b:
                    # beta_j == 0 and theta_j == 0
                    pass
                else:
                    beta_j_hat = (n / pt.norm(x[:, j], 2) ** 2) * soft_thres(x[:, j] @ r_min_j / n, (1 - alpha) * lam)

                    cond_19 = pt.norm(soft_thres(w_j.t() @ (r_min_j - x[:, j] * beta_j_hat) / n, alpha * lam), 2)
                    cond_19 = cond_19 <= (1 - alpha) * lam

                    if cond_19:
                        # beta_j != 0 and theta_j == 0
                        beta[j] = beta_j_hat
                    else:
                        # beta_j != 0 and theta_j != 0
                        pc_mse, pc_penalty_1, pc_penalty_2, pc_penalty_3 = penalties_min_j(
                            beta_0, theta_0, beta, theta, x, z, y, j
                        )
                        objective_prev = partial_objective(
                            beta[j], theta[j, :],
                            x[:, j], z, r_min_j, w_j,
                            alpha, lam,
                            pc_mse.clone(), pc_penalty_1.clone(), pc_penalty_2.clone(), pc_penalty_3.clone()
                        )
                        for _ in range(100):  # Max number of ABG steps
                            r = r_min_j - model_j(beta[j], theta[j, :], x[:, j], w_j)

                            grad_beta_j = -pt.sum(x[:, j] * r) / n
                            grad_theta_j = (-w_j.t() @ r) / n

                            # Solve ABG
                            for l in range(8):
                                tt = t * 0.5 ** l
                                beta_j_hat, theta_j_hat, is_converged = solve_abg(
                                    beta[j], theta[j, :],
                                    grad_beta_j, grad_theta_j,
                                    alpha, lam, tt
                                )
                                if is_converged:
                                    break
                                else:
                                    print('Solve ABG failed @ tt =', tt)

                            # Update coefficients
                            beta[j] = beta_j_hat
                            theta[j, :] = theta_j_hat

                            objective_current = partial_objective(
                                beta[j], theta[j, :],
                                x[:, j], z,
                                r_min_j, w_j,
                                alpha, lam,
                                pc_mse.clone(), pc_penalty_1.clone(), pc_penalty_2.clone(), pc_penalty_3.clone()
                            )
                            improvement = objective_prev - objective_current
                            if pt.abs(improvement) < tolerance:
                                break  # Converged
                            else:
                                objective_prev = objective_current

            # End of parameter sweep
            iter_current_score = objective(beta_0, theta_0, beta, theta, x, z, y, alpha, lam)
            if abs(iter_prev_score - iter_current_score) < tolerance:
                break  # Converged on lam_i

        n_interaction_terms = pt.nonzero(theta).numel()
        if verbose:
            print(
                'Lam:', nth_lam + 1,
                '| N passes:', i+1,
                '| N betas:', pt.nonzero(beta).numel(),
                '| N interaction terms:', n_interaction_terms
            )
        if n_interaction_terms >= max_interaction_terms:  # Maximum interaction terms
            print('Maximum Interaction Terms reached.')
            break

        # Save solution
        lam_list.append(lam.numpy().copy())
        beta_0_list.append(beta_0.detach().numpy().copy())
        theta_0_list.append(theta_0.detach().numpy().copy())
        beta_list.append(beta.detach().numpy().copy())
        theta_list.append(theta.detach().numpy().copy())

    # Return results
    return lam_list, beta_0_list, theta_0_list, beta_list, theta_list
