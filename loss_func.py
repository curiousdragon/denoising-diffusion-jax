import jax.numpy as jnp
from jax import random
from jax import jit

def mean_loss(sigma_t, alpha_ts, mu_pred, x_0, x_t):
    error = 0
    alpha_bar_t = jnp.prod(alpha_ts)
    alpha_t = alpha_ts[-1]
    beta_t = 1 - alpha_t
    alpha_bar_t_minus_1 = alpha_bar_t / alpha_t
    mu_calc = (((alpha_bar_t_minus_1)**0.5)*beta_t/(1 - alpha_bar_t))*x_0 + ((alpha_t**0.5)*(1 - alpha_bar_t_minus_1)/(1 - alpha_bar_t))*x_t
    norm_diff = jnp.linalg.norm(mu_pred - mu_calc)
    error = (1 / (2*sigma_t**2))*norm_diff**2
    return error

def noise_loss(sigma_t, alpha_ts, eps_pred, x_0, x_t, eps):
    error = 0
    alpha_bar_t = jnp.prod(alpha_ts)
    alpha_t = alpha_ts[-1]
    beta_t = 1 - alpha_t
    coeff = (beta_t**2) / (2 * sigma_t**2 * alpha_t * (1 - alpha_bar_t))
    D = len(eps_pred)
    norm_diff = jnp.linalg.norm(eps - eps_pred)
    error = coeff * norm_diff **2
    return error
