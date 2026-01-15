# Abstract Bijectors

::: distreqx.bijectors.AbstractBijector
    options:
        members:
            - __init__
            - forward
            - inverse
            - forward_log_det_jacobian
            - inverse_log_det_jacobian
            - forward_and_log_det
            - inverse_and_log_det
            - same_as

::: distreqx.bijectors.AbstractInvLogDetJacBijector
    options:
        members:
            - inverse_log_det_jacobian

::: distreqx.bijectors.AbstractFwdLogDetJacBijector
    options:
        members:
            - forward_log_det_jacobian

::: distreqx.bijectors.AbstractFowardInverseBijector
    options:
        members:
            - forward
            - inverse

::: distreqx.bijectors.AbstractLinearBijector
    options:
        members:
            - __init__
            - matrix
---