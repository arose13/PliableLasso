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

    # Fit model
    model = PliableLasso()

    print('=== Fitting Model ===')
    start_time = time()
    model.fit(x, z, y)
    stop_time = time()
    print(f'Runtime : {stop_time - start_time:.5f} sec')

    print(f'Rsq = {r2_score(y, model.predict(x, z)):.2%}')

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
    graph.plot(y, model.predict(x, z), 'o', alpha=0.75)
    graph.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='black')
    graph.xlabel('True')
    graph.ylabel('Predicted')
    graph.show()

    print('--- Done ---')
