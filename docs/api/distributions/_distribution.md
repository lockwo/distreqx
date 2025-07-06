# Abstract Distributions

::: distreqx.distributions._distribution.AbstractDistribution
    selection:
        members:
            - _sample_n_and_log_prob
            - log_prob
            - prob
            - cdf
            - survival_function
            - log_survival_function
            - kl_divergence
            - cross_entropy

::: distreqx.distributions._distribution.AbstractSampleLogProbDistribution
    selection:
        members:
            - _sample_n_and_log_prob

::: distreqx.distributions._distribution.AbstractProbDistribution
    selection:
        members:
            - prob

::: distreqx.distributions._distribution.AbstractCDFDistribution
    selection:
        members:
            - cdf

::: distreqx.distributions._distribution.AbstractSTDDistribution
    selection:
        members:
            - stddev

::: distreqx.distributions._distribution.AbstractSurvivalDistribution
    selection:
        members:
            - survival_function
            - log_survival_function

::: distreqx.distributions.transformed.AbstractTransformed
    selection:
        members:
            - distribution
            - bijector
            - dtype
            - event_shape
            - log_prob
            - sample
            - sample_and_log_prob
            - entropy

::: distreqx.distributions.mvn_from_bijector.AbstractMultivariateNormalFromBijector
    selection:
        members:
            - scale
            - loc
            - covariance
            - variance
            - stddev
            - kl_divergence
            
---