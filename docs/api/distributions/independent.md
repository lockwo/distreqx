# Independent Distribution

!!! tip

    `Independent` reinterprets batch dimensions as event dimensions. This is useful when you want to model a multivariate distribution as independent univariate distributions (e.g., diagonal Gaussian) but still want `log_prob` to return a single scalar per sample.

::: distreqx.distributions.Independent
    options:
        members:
            - __init__
---