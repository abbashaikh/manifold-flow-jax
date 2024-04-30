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
    
    def _transform_z_to_x(self, z_phi, z_eps):
        r = 1.0 + z_eps[:, 0]
        a = np.concatenate((2 * np.pi * np.ones((z_phi.shape[0], 1)), z_phi), axis=1)  # n entries, each (2 pi, z_sub)
        sins = np.sin(a)
        sins[:, 0] = 1
        sins = np.cumprod(sins, axis=1)  # n entries, each (1, sin(z0), sin(z1), ..., sin(zk))
        coss = np.cos(a)
        coss = np.roll(coss, -1, axis=1)  # n entries, each (cos(z0), cos(z1), ..., cos(zk), 1)
        exact_sphere = sins * coss  # (n, k+1)
        fuzzy_sphere = exact_sphere * r[:, np.newaxis]
        x = np.concatenate((fuzzy_sphere, z_eps[:, 1:]), axis=1)
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
        z_eps = r - 1
        return z_phi, z_eps