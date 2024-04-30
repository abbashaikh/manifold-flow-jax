import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

@partial(vmap, in_axes=(0,None), out_axes=0)
@partial(jit, static_argnums=(1,))
def _transform_x_to_z(x, latent_dim=1):
    z_phi = jnp.zeros(latent_dim)
    for i in range(latent_dim):
        z_phi = z_phi.at[i].set(jnp.arccos(x[i] / jnp.sum(x[i : latent_dim + 1] ** 2) ** 0.5))
    # Special case for last component, see https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    z_phi = z_phi.at[latent_dim - 1].set(jnp.where(x[latent_dim] < 0.0, 2.0 * jnp.pi - z_phi[latent_dim - 1], z_phi[latent_dim - 1]))

    r = jnp.sum(x[: latent_dim + 1] ** 2) ** 0.5
    z_eps = r - 1
    return z_phi, z_eps

if __name__=="__main__":
      theta = np.linspace(0, 2*np.pi, num=10)
      xs = np.array([np.cos(theta), np.sin(theta)]).T
    #   latent_dim=1
    #   print(xs.shape)
      zs_phi, zs_eps = _transform_x_to_z(xs, 1)
      print(zs_phi)
      print(zs_eps)
    #   print(theta)