#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
import logging
# jax imports
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial

from .base import BaseSimulator

logger = logging.getLogger(__name__)

class SphericalGaussianSimulator(BaseSimulator):
    def __init__(self, latent_dim=8, data_dim=9, phases=0.5 * np.pi, widths=0.25 * np.pi, epsilon=0.01):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._phases = phases * jnp.ones(latent_dim) if isinstance(phases, float) else phases
        self._widths = widths * jnp.ones(latent_dim) if isinstance(widths, float) else widths
        self._epsilon = epsilon

        assert data_dim > latent_dim
        assert epsilon > 0.0

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def parameter_dim(self):
        return None

    @partial(vmap, in_axes=(None, 0, 0), out_axes=0)
    @partial(jit, static_argnums=(0,))
    def _transform_z_to_x(self, z_phi, z_eps):
        r = 1.0 + z_eps[0]
        a = jnp.concatenate((jnp.array([2 * jnp.pi]), z_phi))
        sins = jnp.sin(a)
        sins.at[0].set(1)
        prod_sins = jnp.cumprod(sins)   # (1, sin(z0), sin(z0)*sin(z1), ...)
        coss = jnp.roll(jnp.cos(a), -1) # (cos(z0), cos(z1), ..., cos(zk), 1)
        exact_sphere = prod_sins * coss
        fuzzy_sphere = r * exact_sphere
        x = jnp.concatenate((fuzzy_sphere, z_eps[1:]))
        return x
    
    @partial(vmap, in_axes=(None, 0), out_axes=0)
    @partial(jit, static_argnums=(0,))
    def _transform_x_to_z(self, x):
        z_phi = jnp.zeros(self._latent_dim)
        for i in range(self._latent_dim):
            z_phi = z_phi.at[i].set(jnp.arccos(x[i] / jnp.sum(x[i : self._latent_dim + 1] ** 2) ** 0.5))
        # Special case for last component, see https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        z_phi = z_phi.at[self._latent_dim - 1].set(jnp.where(x[self._latent_dim] < 0.0, 2.0 * jnp.pi - z_phi[self._latent_dim - 1], z_phi[self._latent_dim - 1]))

        r = jnp.sum(x[: self._latent_dim + 1] ** 2) ** 0.5
        z_eps = jnp.copy(x[self._latent_dim :])
        z_eps.at[0].set(r - 1)
        return z_phi, z_eps