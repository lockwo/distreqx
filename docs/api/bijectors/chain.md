# Chain Bijector

!!! warning "Bijectors are applied in reverse order"

    Given a sequence `[f, g]`, the `Chain` bijector computes `f(g(x))`, not `g(f(x))`. This matches the mathematical convention for function composition but may be counterintuitive when building layers sequentially.

::: distreqx.bijectors.Chain
    options:
        members:
            - __init__
---