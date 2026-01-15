# Transformed

!!! tip

    `Transformed` is the foundation for building normalizing flows. Chain together multiple bijectors using [`Chain`](../bijectors/chain.md) to create complex transformations.

!!! warning

    Computing `entropy`, `mean`, and `mode` only works when the bijector has a constant Jacobian determinant. For bijectors with non-constant Jacobians (e.g., neural network-based flows), these methods will raise `NotImplementedError`.

::: distreqx.distributions.Transformed
    options:
        members:
            - __init__
            - mean
            - mode
            - kl_divergence
---