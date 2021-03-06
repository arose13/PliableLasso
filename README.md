plasso
===============================

**NOTE: This project is in the alpha stage while the API and performance is improved**

Author: Stephen Anthony Rose

Original paper: [https://arxiv.org/abs/1712.00484]

```
pip install plasso
```

Abstract
--------

We propose a generalization of the lasso that allows the model coefficients to vary as a function of a general set of modifying variables. These modifiers might be variables such as gender, age or time. The paradigm is quite general, with each lasso coefficient modified by a sparse linear function of the modifying variables Z. The model is estimated in a hierarchical fashion to control the degrees of freedom and avoid overfitting. The modifying variables may be observed, observed only in the training set, or unobserved overall. There are connections of our proposal to varying coefficient models and high-dimensional interaction models. We present a computationally efficient algorithm for its optimization, with exact screening rules to facilitate application to large numbers of predictors. The method is illustrated on a number of different simulated and real examples.

Example
-------

```python
from plasso import PliableLasso

# Input data looks just like sklearn except an extra matrix Z
y = target_data()
x = data()  # Main effects data
z = modifying_data()  # Data used to modify the estimate coefficients for X

# Fit model
model = PliableLasso()
model.fit(x, z, y)

# Cool things to do afterwards
y_hat = model.predict(x, z)

```

Check out the `example.py` file to see more ways to use it.

Installation / Usage
--------------------

To install use pip:

    $ pip install plasso


Or clone the repo:

    $ git clone https://github.com/arose13/PliableLasso.git
    $ python setup.py install
    
Also for large data sets I recommend install `pytorch`
    
Contributing
------------

Me (Stephen Anthony Rose)

If you want to add and improve things be my guest.
