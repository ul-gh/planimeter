# -*- coding: utf-8 -*-
# Function with a sharp peak on a smooth background
import numpy as np
import matplotlib.pyplot as plt
from adaptive_sampling import sample_function

def func(x):
    a = 0.001
    return x + a**2/(a**2 + x**2)

x, y = sample_function(func, [-1, 1], tol=0.03, min_points=6)

xx = np.linspace(-1, 1, 12000)

plt.plot(xx, func(xx), '-', x, y[0], '.')

plt.show()