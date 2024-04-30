import sys
import jax.numpy as jnp
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
