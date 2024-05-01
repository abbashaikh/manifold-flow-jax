import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from jax import make_jaxpr
from jax import random
import jax.scipy.stats.multivariate_normal as norm

latent_dim = 1
data_dim = 2
epsilon = 0.01
phases = 0.5 * np.pi * jnp.ones(latent_dim)
widths = 0.25 * np.pi * jnp.ones(latent_dim)

@partial(vmap, in_axes=(0, 0), out_axes=0)
def _log_density(z_phi, z_eps):
    r = 1.0 + z_eps[0]
    phases_ = phases
    cov_ = jnp.diag(widths)

    ''' add shifted pdfs of the latent variable z_phi '''
    p_sub = norm.pdf(z_phi, mean=phases_, cov=cov_).reshape(latent_dim,)
    # # Variations of polar angles
    # for dim in range(latent_dim - 1):
    #     z = jnp.copy(z_phi)
    #     z = z.at[dim].set(-z_phi[dim])
    #     p_sub = p_sub.at[:].set(p_sub + norm.pdf(z, mean=phases_, cov=cov_))

    #     z = jnp.copy(z_phi)
    #     z = z.at[dim].set(2.0 * np.pi - z_phi[dim])
    #     p_sub = p_sub.at[:].set(p_sub + norm.pdf(z, mean=phases_, cov=cov_))
    # # Variations of aximuthal angle
    z = jnp.copy(z_phi)
    z = z.at[-1].set(-2.0 * jnp.pi + z_phi[-1])
    p_sub = p_sub.at[:].set(p_sub + norm.pdf(z, mean=phases_, cov=cov_))

    # z = jnp.copy(z_phi)
    # z = z.at[-1].set(2.0 * jnp.pi + z_phi[-1])
    # p_sub = p_sub.at[:].set(p_sub + norm.pdf(z, mean=phases_, cov=cov_))

    # ''' find transformed log density from z to x '''
    # logp_sub = jnp.log(p_sub)
    # logp_eps = jnp.log(norm.pdf(z_eps, mean=0.0, cov=epsilon))

    # log_det = latent_dim * jnp.log(jnp.abs(r))
    # log_det = log_det.at[:].set(log_det + jnp.sum(jnp.arange(latent_dim - 1, -1, -1)[jnp.newaxis, :] * jnp.log(jnp.abs(jnp.sin(z_phi)))))

    # logp = jnp.sum(logp_sub) + jnp.sum(logp_eps) + log_det
    return p_sub

if __name__=="__main__":
    num_samples = 10
    z_phi = jnp.linspace(1e-6, 2*jnp.pi + 1e-6, num=num_samples).reshape(num_samples, latent_dim)
    key = random.key(42)
    z_eps = random.multivariate_normal(key, jnp.array([0.0]), jnp.array([epsilon]).reshape(data_dim - latent_dim, data_dim - latent_dim), shape=(num_samples,))
    p_eps = norm.pdf(z_eps[0], mean=0.0, cov=epsilon)
    print(_log_density(z_phi, z_eps).shape)