# Abstract Bijectors

::: distreqx.bijectors._bijector.AbstractBijector
    selection:
        members:
            - __init__
            - forward
            - inverse
            - forward_log_det_jacobian
            - inverse_log_det_jacobian
            - forward_and_log_det
            - inverse_and_log_det
            - same_as

::: distreqx.bijectors._bijector.AbstractInvLogDetJacBijector
    selection:
        members:
            - inverse_log_det_jacobian

::: distreqx.bijectors._bijector.AbstractFwdLogDetJacBijector
    selection:
        members:
            - forward_log_det_jacobian

::: distreqx.bijectors._bijector.AbstractFowardInverseBijector
    selection:
        members:
            - forward
            - inverse

::: distreqx.bijectors._linear.AbstractLinearBijector
    selection:
        members:
            - __init__
            - matrix
---