import numpy as np
from scipy import stats
from plasso import PliableLasso


if __name__ == '__main__':
    # Setup
    n = 100
    p = 50
    k = 4

    beta = np.zeros(p)
    beta[:4] = [2, -2, 2, 2]

    z = stats.bernoulli(p=0.5).rvs(size=(n, k))
    print(z.shape)

    coefs = z.copy()
    coefs[:, 0] = beta[0]
    coefs[:, 1] = beta[1]
    coefs[:, 2] = beta[2] + 2 * z[:, 0]
    coefs[:, 3] = beta[3] * (1 - 2 * z[:, 1])

    x = stats.norm().rvs(size=(n, p))
    print(x.shape)

    y = np.diag(x[:, :4] @ coefs.T) + stats.norm().rvs(n)
    print(y.shape)

    # Model
    plasso = PliableLasso()
    plasso.fit(x, z, y)
