import jax
import functools
from jax import scipy as sp
from jax import numpy as jnp
import numpy as np
from neural_tangents import stax
import neural_tangents as nt

def batched_lstsq_cpu(lhs, rhs, batch_size):
    lhs = np.array(lhs)
    rhs = np.array(rhs)
    ret = []
    for i in range(0, rhs.shape[1], batch_size):
        rhs_chunk = rhs[:, i:i+batch_size]
        ret_chunk = np.linalg.lstsq(lhs, rhs_chunk, rcond=None)[0]
        ret.append(ret_chunk)
        print(f"{i}/{rhs.shape[1]}")
    return jnp.asarray(np.concatenate(ret, axis=1))

def batched_lstsq(lhs, rhs, batch_size):
    u, s, vt = jnp.linalg.svd(lhs, full_matrices=False)
    n2 = rhs.shape[1]
    ret = []
    for i in range(0, n2, batch_size):
        rhs_chunk = rhs[:, i:i+batch_size]
        ret_chunk = vt.T @ ( (u.T @ rhs_chunk) / s[:, None] )
        ret.append(ret_chunk)
        print(f"{i}/{n2}")
    return jnp.concatenate(ret, axis=1)


def make_kernelized_rr_forward(hyper_params):
    _, _, kernel_fn = FullyConnectedNetwork(
        depth=hyper_params["depth"], num_classes=hyper_params["num_items"]
    )
    kernel_fn = functools.partial(kernel_fn, get="ntk")
    if False: #hyper_params["batching"]: 
        kernel_fn = nt.batch(kernel_fn, batch_size=hyper_params["train_batch_size"])


    @jax.jit
    def kernelized_rr_forward(X_train, X_predict, reg=0.1):
        X_train = jnp.asarray(X_train)
        X_predict = jnp.asarray(X_predict)
        print(X_train.shape)
        print(X_predict.shape)
        batch_size = hyper_params["train_batch_size"]
        K_train = kernel_fn(X_train, X_train)
        print("first kernel eval")
        K_predict = kernel_fn(X_predict, X_train)
        print("second kernel eval")
        K_reg = (
            K_train
            + jnp.abs(reg)
            * jnp.trace(K_train)
            * jnp.eye(K_train.shape[0])
            / K_train.shape[0]
        )
        print("rr rhs")
        # Try using jax.numpy.linalg.solve instead of scipy
        if False: #hyper_params["batching"]:
            print("hello")
            try:
                solution = batched_lstsq(K_reg, X_train, batch_size)
            except:
                # cpu fallback
                solution = batched_lstsq_cpu(K_reg, X_train, batch_size)
            print("rr solve")
        else:
            try:
                solution = jnp.linalg.solve(K_reg, X_train, assume_a="pos")
                print("rr solve 1")
            except:
                # Fallback to a more stable but slower method
                print(K_reg.shape)
                print(X_train.shape)
                solution = jnp.linalg.lstsq(K_reg, X_train)[0]
                print("rr solve 2")
        # return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, sym_pos=True))
        if False: #hyper_params["batching"]:
            n1, n2 = K_predict.shape
            n2, n3 = solution.shape
            lhs = K_predict.reshape(4, n1 // 4, n2)
            rhs = solution.reshape(4, n2, n3 // 4)
            ret = jax.lax.dot_general(lhs, rhs, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
        else: 
            ret = jnp.dot(K_predict, solution) 
        print(ret.shape)
        print("finished?")
        return ret
        # return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, assume_a='pos'))

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
