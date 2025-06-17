import jax
import functools
from jax import scipy as sp
from jax import numpy as jnp
from neural_tangents import stax


def make_kernelized_rr_forward(hyper_params):
    _, _, kernel_fn = FullyConnectedNetwork(
        depth=hyper_params["depth"], num_classes=hyper_params["num_items"]
    )
    # NOTE: Un-comment this if the dataset size is very big (didn't need it for experiments in the paper)
    # kernel_fn = nt.batch(kernel_fn, batch_size=128)
    kernel_fn = functools.partial(kernel_fn, get="ntk")

    @jax.jit
    def kernelized_rr_forward(X_train, X_predict, reg=0.1, alpha=None):
        if alpha is None:
            # Compute NTK matrix of the training data. Shape: (U, U)
            K_train = kernel_fn(X_train, X_train)

            # Compute NTK matrix describing the similarity between new users and train users. Shape: (Z, U)
            K_predict = kernel_fn(X_predict, X_train)

            # Instead of regularizing by `reg * I`, regularize by `reg * avg(diag(K)) * I`
            K_diag_avg = jnp.trace(K_train) / K_train.shape[0]
            K_reg = K_train + jnp.abs(reg) * K_diag_avg * jnp.eye(K_train.shape[0]) 

            # Compute dual variables alpha. Shape: (U, I)
            try:
                solution = jnp.linalg.solve(K_reg, X_train, assume_a="pos")
            except:
                # Fallback to a more stable but slower method
                solution = jnp.linalg.lstsq(K_reg, X_train)[0]
            # Compute weighted sum of new users similarity to train users. Shape: (Z, I)
            return jnp.dot(K_predict, solution)
        else:
            K_predict = kernel_fn(X_predict, X_train)
            return jnp.dot(K_predict, alpha)

    return kernelized_rr_forward, kernel_fn


def FullyConnectedNetwork(
    depth, W_std=2**0.5, b_std=0.1, num_classes=10, parameterization="ntk"
):
    activation_fn = stax.Relu()
    dense = functools.partial(
        stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization
    )

    layers = [stax.Flatten()]
    # NOTE: setting width = 1024 doesn't matter as the NTK parameterization will stretch this till \infty
    for _ in range(depth):
        layers += [dense(1024), activation_fn]
    layers += [
        stax.Dense(
            num_classes, W_std=W_std, b_std=b_std, parameterization=parameterization
        )
    ]

    return stax.serial(*layers)
