<h1 align='center'>distreqx</h1>
<h2 align='center'>Distrax + Equinox = distreqx. Easy Pytree probability distributions and bijectors.</h2>

distreqx (pronounced "dist-rex") is a [JAX](https://github.com/google/jax)-based library providing implementations of distributions, bijectors, and tools for statistical and probabilistic machine learning with all benefits of jax (native GPU/TPU acceleration, differentiability, vectorization, distributing workloads, XLA compilation, etc.).

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

key = jax.random.PRNGKey(1234)
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

## Citation

If you found this library useful in academic research, please cite: 

```bibtex
@software{lockwood2024distreqx,
  title = {distreqx: Distributions and Bijectors in Jax},
  author = {Owen Lockwood},
  url = {https://github.com/lockwo/distreqx},
  doi = {[tbd]},
}
```

(Also consider starring the project on GitHub.)

## See also: other libraries in the JAX ecosystem

[GPJax](https://github.com/JaxGaussianProcesses/GPJax): Gaussian processes in JAX. 

[flowjax](https://github.com/danielward27/flowjax): Normalizing flows in JAX.

[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.  

[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  

[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  

[diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers in JAX. Autodifferentiable and GPU-capable.

[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  

## Original distrax copyright

```
Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
```
