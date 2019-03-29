import numpy as np
from time import time
from scipy import stats
from plasso import PliableLasso
import matplotlib.pyplot as graph
from sklearn.metrics import r2_score


if __name__ == '__main__':
    # Setup
    np.random.seed(1992)
    n = 200
    p = 10
    k = 5

    x = stats.norm().rvs((n, p))
    z = stats.norm().rvs((n, k))

    y = 4*x[:, 1] + 5*x[:, 1] * z[:, 3]
    y += 3*stats.norm().rvs(n)
    y += 5

    # Fit model
    model = PliableLasso(cv=3, verbose=True, eps=1e-2, normalize=True)

    print('=== Fitting Model ===')
    start_time = time()
    model.fit(x, z, y)
    stop_time = time()
    print(f'Runtime : {stop_time - start_time:.5f} sec')
    y_hat = model.predict(x, z)
    print(f'Rsq = {r2_score(y, y_hat):.2%}')

    print('beta_0')
    print(model.beta_0)
    print('theta_0')
    print(model.theta_0)
    print('beta')
    print(model.beta[np.abs(model.beta) > 2])
    print('theta')
    print(model.theta[np.abs(model.theta) > 2])

    print()

    # Plot coefficient paths
    model.plot_coef_paths()
    graph.show()

    model.plot_interaction_paths()
    graph.show()

    model.plot_intercepts_path()
    graph.show()

    model.plot_score_path()
    graph.show()

    graph.figure(figsize=(6, 6))
    graph.plot(y, y_hat, 'o', alpha=0.75)
    graph.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='black')
    graph.xlabel('True')
    graph.ylabel('Predicted')
    graph.show()

    print('--- Done ---')
