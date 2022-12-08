import jax.numpy as jnp
from jax import Array
from jax.config import config

config.update("jax_array", True)

import matplotlib.pyplot as plt


def show_plots(states: Array, standard=False, description=None):
    n = len(states)
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(9, 2))
    for i in range(n):
        if standard:
            axs[i].imshow(states[i], vmin=-1, vmax=2)
        else:
            axs[i].imshow(states[i])
    if description:
        plt.suptitle(description)
    plt.show()


def run_tests(inputs, expected, function, test_str):
    results = []
    for test_inputs, test_expected in zip(inputs, expected):
        actual_output = function(*[jnp.array(i) for i in test_inputs])
        result = jnp.abs(jnp.array(test_expected) - jnp.array(actual_output))
        results.append(result)

    average_result = jnp.mean(jnp.array(results))
    print(
        f"{test_str}: Average difference between expected and actual: {average_result}"
    )

    threshold = 1e-5
    if average_result < threshold:
        print(f"{test_str}: Result vs threshold: {average_result} < {threshold}")
        print(f"{test_str} test passed! :)")
        print()
    else:
        print(f"{test_str}: Result vs threshold: {average_result} => {threshold}.")
        print(f"{test_str} test failed :(")
        print()

    return average_result
