# distreqx

distreqx (pronounced "dist-rex") is a [JAX](https://github.com/google/jax)-based library providing implementations of distributions, bijectors, and tools for statistical and probabilistic machine learning with all benefits of jax (native GPU/TPU acceleration, differentiability, vectorization, distributing workloads, XLA compilation, etc.).

!!! tip

    New to distreqx? Start with the [Examples](examples/01_vae.ipynb) to see distributions and bijectors in action, then explore the API reference for details.

The origin of this package is a reimplementation of [distrax](https://github.com/google-deepmind/distrax), (which is a subset of [TensorFlow Probability (TFP)](https://github.com/tensorflow/probability), with some new features and emphasis on jax compatibility) using [equinox](https://github.com/patrick-kidger/equinox). As a result, much of the original code/comments/documentation/tests are directly taken or adapted from distrax (original distrax copyright available at end of README.)

Current features include:

- Probability distributions
- Bijectors


## Installation

```
pip install distreqx
```

or

```
git clone https://github.com/lockwo/distreqx.git
cd distreqx
pip install -e .
```

Requires Python 3.9+, JAX 0.4.11+, and [Equinox](https://github.com/patrick-kidger/equinox) 0.11.0+.

## Documentation

Available at https://lockwo.github.io/distreqx/.

## Quick example

```python
import jax
from jax import numpy as jnp
from distreqx import distributions

key = jax.random.key(1234)
mu = jnp.array([-1., 0., 1.])
sigma = jnp.array([0.1, 0.2, 0.3])

dist = distributions.MultivariateNormalDiag(mu, sigma)

samples = dist.sample(key)

print(dist.log_prob(samples))
```

## Differences with Distrax

- No official support/interoperability with TFP
- The concept of a batch dimension is dropped. If you want to operate on a batch, use `vmap` (note, this can be used in construction as well, e.g. [vmaping the construction](https://docs.kidger.site/equinox/tricks/#ensembling) of a `ScalarAffine`)
- Broader pytree enablement 
- Strict [abstract/final](https://docs.kidger.site/equinox/pattern/) design pattern