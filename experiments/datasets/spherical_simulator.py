#! /usr/bin/env python

import numpy as np
import logging
# jax imports
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
from jax import random
import jax.scipy.stats.multivariate_normal as norm

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
        self._key = random.key(42)

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
    
    def log_density(self, x, parameters=None, precise=False):
        z_phi, z_eps = self._transform_x_to_z(x)
        logp = self._log_density(z_phi, z_eps, precise=precise)
        return logp

    def sample(self, n, parameters=None):
        z_phi, z_eps = self._draw_z(jnp.arange(n))
        x = self._transform_z_to_x(z_phi, z_eps)
        return x

    def sample_ood(self, n, parameters=None):
        z_phi, _ = self._draw_z(jnp.arange(n))
        z_eps = np.random.uniform(-3.0 * self._epsilon, 0.0, size=(n, self._data_dim - self._latent_dim))
        x = self._transform_z_to_x(z_phi, z_eps)
        return x

    def distance_from_manifold(self, x):
        z_phi, z_eps = self._transform_x_to_z(x)
        return np.sum(z_eps ** 2, axis=1) ** 0.5

    @partial(vmap, in_axes=(None, 0), out_axes=0)
    @partial(jit, static_argnums=(0,))
    def _draw_z(self, n):
        # Spherical coordinates
        means_ = self._phases
        cov_ = jnp.diag(self._widths ** 2)

        self._key, key1, key2 = random.split(self._key, 3)
        z_phi = random.multivariate_normal(key1, mean=means_, cov=cov_)
        z_phi = jnp.mod(z_phi, 2.0 * jnp.pi)

        # Fuzzy coordinates
        z_eps = random.normal(key2, shape=(self._data_dim - self._latent_dim,)) * self._epsilon
        return z_phi, z_eps

    @partial(vmap, in_axes=(None, 0, 0), out_axes=0)
    @partial(jit, static_argnums=(0,))
    def _transform_z_to_x(self, z_phi, z_eps):
        r = 1.0 + z_eps[0]
        a = jnp.concatenate((jnp.array([2 * jnp.pi]), z_phi))
        sins = jnp.sin(a)
        sins = sins.at[0].set(1)
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
    
    @partial(vmap, in_axes=(None, 0, 0), out_axes=0)
    @partial(jit, static_argnums=(0,))
    def _log_density(self, z_phi, z_eps):
        r = 1.0 + z_eps[0]
        means_ = self._phases
        cov_ = jnp.diag(self._widths ** 2)

        ''' add shifted pdfs of the latent variable z_phi '''
        p_sub = norm.pdf(z_phi, mean=means_, cov=cov_).reshape(self._latent_dim,)
        # Variations of polar angles
        for dim in range(self._latent_dim - 1):
            z = jnp.copy(z_phi)
            z = z.at[dim].set(-z_phi[dim])
            p_sub = p_sub.at[:].set(p_sub + norm.pdf(z, mean=means_, cov=cov_))

            z = jnp.copy(z_phi)
            z = z.at[dim].set(2.0 * np.pi - z_phi[dim])
            p_sub = p_sub.at[:].set(p_sub + norm.pdf(z, mean=means_, cov=cov_))
        # Variations of aximuthal angle
        z = jnp.copy(z_phi)
        z = z.at[-1].set(-2.0 * jnp.pi + z_phi[-1])
        p_sub = p_sub.at[:].set(p_sub + norm.pdf(z, mean=means_, cov=cov_))

        z = jnp.copy(z_phi)
        z = z.at[-1].set(2.0 * jnp.pi + z_phi[-1])
        p_sub = p_sub.at[:].set(p_sub + norm.pdf(z, mean=means_, cov=cov_))

        ''' find transformed log density from z to x '''
        logp_sub = jnp.log(p_sub)
        logp_eps = jnp.log(norm.pdf(z_eps, mean=0.0, cov=self._epsilon ** 2))

        log_det = self._latent_dim * jnp.log(jnp.abs(r)) + jnp.sum(jnp.arange(self._latent_dim - 1, -1, -1) * jnp.log(jnp.abs(jnp.sin(z_phi))))

        logp = jnp.sum(logp_sub) + jnp.sum(logp_eps) + log_det
        return logp
