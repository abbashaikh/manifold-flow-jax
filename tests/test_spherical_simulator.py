import sys
import jax.numpy as jnp
from jax import random
import numpy as np

from experiments.datasets import SphericalGaussianSimulator

latent_dim = 1
data_dim = 2
epsilon = 0.01
simulator = SphericalGaussianSimulator(latent_dim=latent_dim, data_dim=data_dim, epsilon=epsilon)

def test_transform_x_to_z():
    num_samples = 10
    theta = jnp.linspace(0, 2*jnp.pi, num=num_samples)
    xs = jnp.array([jnp.cos(theta), jnp.sin(theta)]).T
    zs_phi, zs_eps = simulator._transform_x_to_z(xs)
    
    assert zs_phi.shape==(num_samples, latent_dim)
    assert zs_eps.shape==(num_samples, data_dim-latent_dim)

def test_transform_z_to_x():
    num_samples = 10
    z_phi = jnp.linspace(0, 2*jnp.pi, num=num_samples).reshape(num_samples, latent_dim)
    z_eps = jnp.zeros((num_samples, data_dim-latent_dim), dtype=float)
    xs = simulator._transform_z_to_x(z_phi, z_eps)

    assert xs.shape==(num_samples, data_dim)

def test_log_density():
    num_samples = 10
    z_phi = jnp.linspace(1e-6, 2*jnp.pi + 1e-6, num=num_samples).reshape(num_samples, latent_dim)
    key = random.key(42)
    z_eps = random.multivariate_normal(key, jnp.array([0.0]), jnp.array([epsilon]).reshape(data_dim - latent_dim, data_dim - latent_dim), shape=(num_samples,))
    logp = simulator._log_density(z_phi, z_eps)

    assert logp.shape == (num_samples,)
