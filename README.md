<h1 align='center'>distreqx</h1>
<h2 align='center'>Distrax + Equinox = distreqx. Easy Pytree probability distributions and bijectors.</h2>

distreqx is a [JAX](https://github.com/google/jax)-based library providing implementations of a subset of [TensorFlow Probability (TFP)](https://github.com/tensorflow/probability), with some new features and emphasis on jax compatibility.

This is a largely as reimplementation of [distrax](https://github.com/google-deepmind/distrax) using [equinox](https://github.com/patrick-kidger/equinox), much of the code/comments/documentation/tests are directly taken or adapted from distrax so all credit to the DeepMind team.  

Features include:

- Probability distributions
- Bijectors


## Installation

```
pip install distreqx
```

Requires Python 3.9+, JAX 0.4.13+, and [Equinox](https://github.com/patrick-kidger/equinox) 0.11.0+.

## Documentation

Available at .

## Quick example

```python
from distreqx import
```

## Differences with Distrax

- No support for TFP

## Citation

If you found this library useful in academic research, please cite: 

```bibtex
```

(Also consider starring the project on GitHub.)

## See also: other libraries in the JAX ecosystem

[Equinox](https://github.com/patrick-kidger/equinox): neural networks and everything not already in core JAX!  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for shape/dtype of arrays.  
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  
[diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers in JAX. Autodifferentiable and GPU-capable.

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  

## Original distrax copyright

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