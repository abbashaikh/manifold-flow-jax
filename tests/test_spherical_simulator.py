import sys
import jax.numpy as jnp

sys.path.append("../")
from experiments.datasets import SphericalGaussianSimulator

def test_transform_x_to_z():
    epsilon = 0.01
    simulator = SphericalGaussianSimulator(latent_dim=1, data_dim=2, epsilon=epsilon)

    num_samples = 10
    theta = jnp.linspace(0, 2*jnp.pi, num=num_samples)
    xs = jnp.array([jnp.cos(theta), jnp.sin(theta)]).T
    zs_phi, zs_eps = simulator._transform_x_to_z(xs)
    assert zs_phi.shape==(num_samples, 1)
    assert zs_eps.shape==(num_samples,)
