from jax import Array
from jax.config import config

config.update("jax_array", True)

import matplotlib.pyplot as plt


def show_outputs(states: Array):
    n = len(states)
    _, axs = plt.subplots(nrows=1, ncols=n, figsize=(9, 2))
    for i in range(n):
        axs[i].imshow(states[i], vmin=-1, vmax=2)
    plt.show()
