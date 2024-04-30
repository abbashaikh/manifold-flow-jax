import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from jax import make_jaxpr

@partial(vmap, in_axes=(0, 0), out_axes=0)
@partial(jit)
def _transform_z_to_x(z_phi, z_eps):
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

if __name__=="__main__":
    num_samples = 10
    latent_dim = 1
    data_dim = 2

    z_phi = jnp.linspace(0, 2*jnp.pi, num=num_samples).reshape(num_samples, latent_dim)
    z_eps = jnp.zeros((num_samples, data_dim-latent_dim), dtype=float)
    # print(make_jaxpr(_transform_z_to_x)(z_phi, z_eps))
    xs = _transform_z_to_x(z_phi, z_eps)
    print(xs.shape)
# z_phi = jnp.linspace(0, 2*jnp.pi, num=5)
# arr = jnp.concatenate((jnp.array([23]), z_phi))
# print(arr)