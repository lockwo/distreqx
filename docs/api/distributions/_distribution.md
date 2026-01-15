# Abstract Distributions

::: distreqx.distributions.AbstractDistribution
    options:
        members:
            - _sample_n_and_log_prob
            - log_prob
            - prob
            - cdf
            - survival_function
            - log_survival_function
            - kl_divergence
            - cross_entropy

::: distreqx.distributions.AbstractSampleLogProbDistribution
    options:
        members:
            - _sample_n_and_log_prob

::: distreqx.distributions.AbstractProbDistribution
    options:
        members:
            - prob

::: distreqx.distributions.AbstractCDFDistribution
    options:
        members:
            - cdf

::: distreqx.distributions.AbstractSTDDistribution
    options:
        members:
            - stddev

::: distreqx.distributions.AbstractSurvivalDistribution
    options:
        members:
            - survival_function
            - log_survival_function

::: distreqx.distributions.AbstractTransformed
    options:
        members:
            - distribution
            - bijector
            - dtype
            - event_shape
            - log_prob
            - sample
            - sample_and_log_prob
            - entropy

::: distreqx.distributions.AbstractMultivariateNormalFromBijector
    options:
        members:
            - scale
            - loc
            - covariance
            - variance
            - stddev
            - kl_divergence
            
---