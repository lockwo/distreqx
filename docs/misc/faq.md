# FAQ

## Why not just use distrax?

The simple answer to that question is "I tried". Distrax is a the product of a lot of great work, especially helpful for working with TFP, but in the current era of jax packages lacks important elements:

- It's only semi-maintained (there have been no responses to any issues in the last >6 months)
- It doesn't always play nice with other jax packages and can be slow (see: [#193](https://github.com/google-deepmind/distrax/issues/193), [#383](https://github.com/patrick-kidger/diffrax/issues/383), [#252](https://github.com/patrick-kidger/equinox/issues/252), [#269](https://github.com/patrick-kidger/equinox/issues/269), [#16](https://github.com/JaxGaussianProcesses/JaxUtils/issues/16), [#16170](https://github.com/google/jax/issues/16170))
- You need Tensorflow to use it 

## Why use equinox?

The `Jittable` class is basically an equinox module (if you squint) and while we could reimplement a custom Module class (like GPJax does), why reinvent the wheel? Equinox is actively being developed and should it become inactive is still possible to maintain.
