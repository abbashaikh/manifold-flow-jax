import jax.numpy as jnp

def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = shape + x.shape[1:]
    return x.reshape(new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.ndim():
        raise ValueError("Number of leading dims can't be greater than total number of dims.")
    new_shape = (-1,) + x.shape[num_dims:]
    return x.reshape(new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    return jnp.repeat(x, repeats=num_reps, axis=0)


def product(x):
    try:
        prod = 1
        for factor in x:
            prod *= factor
        return prod
    except:
        return x


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0