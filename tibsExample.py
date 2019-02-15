import numpy as np
from time import time
from scipy import stats
from plasso import PliableLasso
import matplotlib.pyplot as graph


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
    print('=== Fitting Model ===')
    start_time = time()
    model = PliableLasso()
    model.fit(x, z, y)
    stop_time = time()
    print(f'Runtime : {stop_time - start_time:.5f} sec')

    # Plot coefficient paths
    model.plot_coef_paths()
    graph.show()

    model.plot_interaction_paths()
    graph.show()

