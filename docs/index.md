# distreqx

distreqx (pronounced "dist-rex") is a [JAX](https://github.com/google/jax)-based library providing implementations of distributions, bijectors, and tools for statistical and probabilistic machine learning with all benefits of jax (native GPU/TPU acceleration, differentiability, vectorization, distributing workloads, XLA compilation, etc.).

!!! tip

    New to distreqx? Start with the [Examples](examples/01_vae.ipynb) to see distributions and bijectors in action, then explore the API reference for details.

The origin of this package is a reimplementation of [distrax](https://github.com/google-deepmind/distrax), (which is a subset of [TensorFlow Probability (TFP)](https://github.com/tensorflow/probability), with some new features and emphasis on jax compatibility) using [equinox](https://github.com/patrick-kidger/equinox). As a result, much of the original code/comments/documentation/tests are directly taken or adapted from distrax (original distrax copyright available at end of README.)


## Why distreqx?

- **Pure JAX** — No TensorFlow dependency. Works seamlessly with the modern JAX ecosystem.
- **Equinox-based** — Distributions and bijectors are pytrees, so they play nicely with `jit`, `vmap`, `grad`, and other JAX transformations out of the box.
- **Actively maintained** — Unlike distrax, which has been largely unmaintained, distreqx receives regular updates and bug fixes.
- **Strict design patterns** — Follows the [abstract/final pattern](https://docs.kidger.site/equinox/pattern/) for clean, extensible code.
- **No batch dimension** — Uses `vmap` for batching, which is more explicit and composable than implicit batch dimensions.


## Features at a Glance

### Distributions

| Distribution | Class |
|---|---|
| Normal (univariate) | `distributions.Normal` |
| Multivariate Normal (diagonal) | `distributions.MultivariateNormalDiag` |
| Multivariate Normal (full covariance) | `distributions.MultivariateNormalFullCovariance` |
| Multivariate Normal (triangular) | `distributions.MultivariateNormalTri` |
| Multivariate Normal (from bijector) | `distributions.MultivariateNormalFromBijector` |
| Bernoulli | `distributions.Bernoulli` |
| Beta | `distributions.Beta` |
| Categorical | `distributions.Categorical` |
| One-Hot Categorical | `distributions.OneHotCategorical` |
| Gamma | `distributions.Gamma` |
| Logistic | `distributions.Logistic` |
| Uniform | `distributions.Uniform` |
| Independent | `distributions.Independent` |
| Empirical | `distributions.Empirical` |
| Transformed | `distributions.Transformed` |
| Mixture (same family) | `distributions.MixtureSameFamily` |

### Bijectors

| Bijector | Class |
|---|---|
| Scalar Affine | `bijectors.ScalarAffine` |
| Shift | `bijectors.Shift` |
| Sigmoid | `bijectors.Sigmoid` |
| Tanh | `bijectors.Tanh` |
| Identity | `bijectors.Identity` |
| Block | `bijectors.Block` |
| Chain | `bijectors.Chain` |
| Inverse | `bijectors.Inverse` |
| Diagonal Linear | `bijectors.DiagLinear` |
| Triangular Linear | `bijectors.TriangularLinear` |
| Rational Quadratic Spline | `bijectors.RationalQuadraticSpline` |


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

Requires Python 3.10+, JAX 0.4.11+, and [Equinox](https://github.com/patrick-kidger/equinox) 0.11.0+.


## Quick Examples

### Distributions

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

### Bijectors

```python
import jax
from jax import numpy as jnp
from distreqx import bijectors

# Create an affine bijector: y = 2x + 1
bijector = bijectors.ScalarAffine(shift=1.0, scale=2.0)

x = jnp.array([0.0, 1.0, 2.0])
y = bijector.forward(x)          # [1., 3., 5.]
x_reconstructed = bijector.inverse(y)  # [0., 1., 2.]

# Log determinant of the Jacobian (useful for normalizing flows)
log_det = bijector.forward_log_det_jacobian(x)
```

### Transformed Distributions

```python
import jax
from jax import numpy as jnp
from distreqx import distributions, bijectors

# Create a log-normal distribution by transforming a normal with exp
normal = distributions.Normal(loc=0.0, scale=1.0)
exp_bijector = bijectors.Chain([bijectors.ScalarAffine(shift=0.0, scale=1.0)])

log_normal = distributions.Transformed(
    distribution=normal,
    bijector=exp_bijector
)

key = jax.random.key(42)
samples = log_normal.sample(key)
log_prob = log_normal.log_prob(samples)
```

### JAX Transformations

All distributions and bijectors work seamlessly with JAX transformations:

```python
import jax
from jax import numpy as jnp
from distreqx import distributions

# vmap over different distribution parameters
mus = jnp.array([0.0, 1.0, 2.0])
sigmas = jnp.array([0.5, 1.0, 1.5])

# Create and sample from multiple distributions at once
keys = jax.random.split(jax.random.key(0), 3)
samples = jax.vmap(
    lambda mu, sigma, key: distributions.Normal(mu, sigma).sample(key)
)(mus, sigmas, keys)

# Gradient through log_prob
def neg_log_prob(mu, x):
    return -distributions.Normal(mu, 1.0).log_prob(x)

grad_fn = jax.grad(neg_log_prob)
print(grad_fn(0.0, 1.0))  # Gradient of NLL w.r.t. mu
```


## Differences with Distrax

- No official support/interoperability with TFP
- The concept of a batch dimension is dropped. If you want to operate on a batch, use `vmap` (note, this can be used in construction as well, e.g. [vmaping the construction](https://docs.kidger.site/equinox/tricks/#ensembling) of a `ScalarAffine`)
- Broader pytree enablement 
- Strict [abstract/final](https://docs.kidger.site/equinox/pattern/) design pattern
