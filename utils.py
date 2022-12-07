from jax import Array
from jax.config import config

config.update("jax_array", True)

import matplotlib.pyplot as plt
from typing import List


def show_outputs(states: Array):
    n = len(states)
    _, axs = plt.subplots(nrows=1, ncols=n, figsize=(9, 2))
    for i in range(n):
        axs[i].imshow(states[i])
    plt.show()


def show_states(state_visuals: List[List[Array]]):
    for states in state_visuals:
        show_outputs(states)
