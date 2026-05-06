import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


def _normalize_bin_sizes(
    unnormalized_bin_sizes: Array, total_size: float, min_bin_size: float
) -> Array:
    """Make bin sizes sum to `total_size` and be no less than `min_bin_size`."""
    num_bins = unnormalized_bin_sizes.shape[-1]
    if num_bins * min_bin_size > total_size:
        raise ValueError(
            f"The number of bins ({num_bins}) times the minimum bin size "
            f"({min_bin_size}) cannot be greater than the "
            f"total bin size ({total_size})."
        )
    bin_sizes = jax.nn.softmax(unnormalized_bin_sizes, axis=-1)
    return bin_sizes * (total_size - num_bins * min_bin_size) + min_bin_size


def _normalize_knot_slopes(
    unnormalized_knot_slopes: Array, min_knot_slope: float
) -> Array:
    """Make knot slopes be no less than `min_knot_slope`."""
    if min_knot_slope >= 1.0:
        raise ValueError(
            f"The minimum knot slope must be less than 1; got {min_knot_slope}."
        )
    min_knot_slope_arr = jnp.array(min_knot_slope, dtype=unnormalized_knot_slopes.dtype)
    offset = jnp.log(jnp.exp(1.0 - min_knot_slope_arr) - 1.0)
    return jax.nn.softplus(unnormalized_knot_slopes + offset) + min_knot_slope_arr


def _rational_quadratic_spline_fwd(
    x: Array, x_pos: Array, y_pos: Array, knot_slopes: Array
) -> tuple[Array, Array]:
    """Applies a rational-quadratic spline to a scalar."""
    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]
    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)

    first_bin = jnp.concatenate(
        [jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]
    ).astype(bool)

    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width

    z = (x - x_pos_bin[0]) / bin_width
    z = jnp.clip(z, 0.0, 1.0)
    sq_z = z * z
    z1mz = z - sq_z
    sq_1mz = (1.0 - z) ** 2
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    numerator = bin_height * (bin_slope * sq_z + knot_slopes_bin[0] * z1mz)
    denominator = bin_slope + slopes_term * z1mz
    y = y_pos_bin[0] + numerator / denominator

    logdet = (
        2.0 * jnp.log(bin_slope)
        + jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        - 2.0 * jnp.log(denominator)
    )

    y = jnp.where(below_range, (x - x_pos[0]) * knot_slopes[0] + y_pos[0], y)
    y = jnp.where(above_range, (x - x_pos[-1]) * knot_slopes[-1] + y_pos[-1], y)
    logdet = jnp.where(below_range, jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, jnp.log(knot_slopes[-1]), logdet)
    return y, logdet


def _safe_quadratic_root(a: Array, b: Array, c: Array) -> Array:
    sqrt_diff = b**2 - 4.0 * a * c
    safe_sqrt = jnp.sqrt(jnp.clip(sqrt_diff, jnp.finfo(sqrt_diff.dtype).tiny))
    safe_sqrt = jnp.where(sqrt_diff > 0.0, safe_sqrt, 0.0)

    numerator_1 = 2.0 * c
    denominator_1 = -b - safe_sqrt
    numerator_2 = -b + safe_sqrt
    denominator_2 = 2 * a

    numerator = jnp.where(b >= 0, numerator_1, numerator_2)
    denominator = jnp.where(b >= 0, denominator_1, denominator_2)
    return numerator / denominator


def _rational_quadratic_spline_inv(
    y: Array, x_pos: Array, y_pos: Array, knot_slopes: Array
) -> tuple[Array, Array]:
    """Applies the inverse of a rational-quadratic spline to a scalar."""
    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]
    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)

    first_bin = jnp.concatenate(
        [jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]
    ).astype(bool)

    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width
    w = (y - y_pos_bin[0]) / bin_height
    w = jnp.clip(w, 0.0, 1.0)

    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    c = -bin_slope * w
    b = knot_slopes_bin[0] - slopes_term * w
    a = bin_slope - b

    z = _safe_quadratic_root(a, b, c)
    z = jnp.clip(z, 0.0, 1.0)
    x = bin_width * z + x_pos_bin[0]

    sq_z = z * z
    z1mz = z - sq_z
    sq_1mz = (1.0 - z) ** 2
    denominator = bin_slope + slopes_term * z1mz
    logdet = (
        -2.0 * jnp.log(bin_slope)
        - jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        + 2.0 * jnp.log(denominator)
    )

    x = jnp.where(below_range, (y - y_pos[0]) / knot_slopes[0] + x_pos[0], x)
    x = jnp.where(above_range, (y - y_pos[-1]) / knot_slopes[-1] + x_pos[-1], x)
    logdet = jnp.where(below_range, -jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, -jnp.log(knot_slopes[-1]), logdet)
    return x, logdet


class RationalQuadraticSpline(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """A rational-quadratic spline bijector.

    Implements the spline bijector introduced by:
    > Durkan et al., Neural Spline Flows, https://arxiv.org/abs/1906.04032, 2019.

    This bijector is a monotonically increasing spline operating on an interval
    [a, b], such that f(a) = a and f(b) = b. Outside the interval [a, b], the
    bijector defaults to a linear transformation whose slope matches that of the
    spline at the nearest boundary (either a or b). The range boundaries a and b
    are hyperparameters passed to the constructor.

    The spline on the interval [a, b] consists of `num_bins` segments, on each of
    which the spline takes the form of a rational quadratic (ratio of two
    quadratic polynomials). The first derivative of the bijector is guaranteed to
    be continuous on the whole real line. The second derivative is generally not
    continuous at the knot points (bin boundaries).

    The spline is parameterized by the bin sizes on the x and y axis, and by the
    slopes at the knot points. All spline parameters are passed to the constructor
    as an unconstrained array `params` of shape `[..., 3 * num_bins + 1]`. The
    spline parameters are extracted from `params`, and are reparameterized
    internally as appropriate. The number of bins is a hyperparameter, and is
    implicitly defined by the last dimension of `params`.

    This bijector is applied elementwise. Given some input `x`, the parameters
    `params` and the input `x` are broadcast against each other. For example,
    suppose `x` is of shape `[N, D]`. Then:
    - If `params` is of shape `[3 * num_bins + 1]`, the same spline is identically
    applied to each element of `x`.
    - If `params` is of shape `[D, 3 * num_bins + 1]`, the same spline is applied
    along the first axis of `x` but a different spline is applied along the
    second axis of `x`.
    - If `params` is of shape `[N, D, 3 * num_bins + 1]`, a different spline is
    applied to each element of `x`.
    - If `params` is of shape `[M, N, D, 3 * num_bins + 1]`, `M` different splines
    are applied to each element of `x`, and the output is of shape `[M, N, D]`.
    """

    num_bins: int = eqx.field(static=True)
    x_pos: Array
    y_pos: Array
    knot_slopes: Array

    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def __init__(
        self,
        params: Array,
        range_min: float,
        range_max: float,
        boundary_slopes: str = "unconstrained",
        min_bin_size: float = 1e-4,
        min_knot_slope: float = 1e-4,
    ):
        """Initializes a RationalQuadraticSpline bijector.

        Args:
        params: array of shape `[..., 3 * num_bins + 1]`, the unconstrained
            parameters of the bijector. The number of bins is implicitly defined by
            the last dimension of `params`. The parameters can take arbitrary
            unconstrained values; the bijector will reparameterize them internally
            and make sure they obey appropriate constraints. If `params` is the
            all-zeros array, the bijector becomes the identity function everywhere
            on the real line.
        range_min: the lower bound of the spline's range. Below `range_min`, the
            bijector defaults to a linear transformation.
        range_max: the upper bound of the spline's range. Above `range_max`, the
            bijector defaults to a linear transformation.
        boundary_slopes: controls the behaviour of the slope of the spline at the
            range boundaries (`range_min` and `range_max`). It is used to enforce
            certain boundary conditions on the spline. Available options are:
            - 'unconstrained': no boundary conditions are imposed; the slopes at the
            boundaries can vary freely.
            - 'lower_identity': the slope of the spline is set equal to 1 at the
            lower boundary (`range_min`). This makes the bijector equal to the
            identity function for values less than `range_min`.
            - 'upper_identity': similar to `lower_identity`, but now the slope of
            the spline is set equal to 1 at the upper boundary (`range_max`). This
            makes the bijector equal to the identity function for values greater
            than `range_max`.
            - 'identity': combines the effects of 'lower_identity' and
            'upper_identity' together. The slope of the spline is set equal to 1
            at both boundaries (`range_min` and `range_max`). This makes the
            bijector equal to the identity function outside the interval
            `[range_min, range_max]`.
            - 'circular': makes the slope at `range_min` and `range_max` be the
            same. This implements the "circular spline" introduced by:
            > Rezende et al., Normalizing Flows on Tori and Spheres,
            > https://arxiv.org/abs/2002.02428, 2020.
            This option should be used when the spline operates on a circle
            parameterized by an angle in the interval `[range_min, range_max]`,
            where `range_min` and `range_max` correspond to the same point on the
            circle.
        min_bin_size: The minimum bin size, in either the x or the y axis. Should
            be a small positive number, chosen for numerical stability. Guarantees
            that no bin in either the x or the y axis will be less than this value.
        min_knot_slope: The minimum slope at each knot point. Should be a small
            positive number, chosen for numerical stability. Guarantess that no knot
            will have a slope less than this value.
        """
        if params.shape[-1] % 3 != 1 or params.shape[-1] < 4:
            raise ValueError(
                f"The last dimension of `params` must have size `3 * num_bins + 1` "
                f"and `num_bins` must be at least 1. Got size {params.shape[-1]}."
            )
        if range_min >= range_max:
            raise ValueError(
                f"`range_min` must be less than `range_max`. Got "
                f"`range_min={range_min}` and `range_max={range_max}`."
            )

        self.num_bins = (params.shape[-1] - 1) // 3
        dtype = params.dtype

        unnormalized_bin_widths = params[..., : self.num_bins]  # noqa: E203
        unnormalized_bin_heights = params[
            ..., self.num_bins : 2 * self.num_bins  # noqa: E203
        ]
        unnormalized_knot_slopes = params[..., 2 * self.num_bins :]  # noqa: E203

        range_size = range_max - range_min
        bin_widths = _normalize_bin_sizes(
            unnormalized_bin_widths, range_size, min_bin_size
        )
        bin_heights = _normalize_bin_sizes(
            unnormalized_bin_heights, range_size, min_bin_size
        )

        x_pos_core = range_min + jnp.cumsum(bin_widths[..., :-1], axis=-1)
        y_pos_core = range_min + jnp.cumsum(bin_heights[..., :-1], axis=-1)

        pad_shape = params.shape[:-1] + (1,)
        pad_below = jnp.full(pad_shape, range_min, dtype=dtype)
        pad_above = jnp.full(pad_shape, range_max, dtype=dtype)

        self.x_pos = jnp.concatenate([pad_below, x_pos_core, pad_above], axis=-1)
        self.y_pos = jnp.concatenate([pad_below, y_pos_core, pad_above], axis=-1)

        knot_slopes_core = _normalize_knot_slopes(
            unnormalized_knot_slopes, min_knot_slope
        )

        if boundary_slopes == "unconstrained":
            self.knot_slopes = knot_slopes_core
        elif boundary_slopes == "lower_identity":
            ones = jnp.ones(pad_shape, dtype)
            self.knot_slopes = jnp.concatenate(
                [ones, knot_slopes_core[..., 1:]], axis=-1
            )
        elif boundary_slopes == "upper_identity":
            ones = jnp.ones(pad_shape, dtype)
            self.knot_slopes = jnp.concatenate(
                [knot_slopes_core[..., :-1], ones], axis=-1
            )
        elif boundary_slopes == "identity":
            ones = jnp.ones(pad_shape, dtype)
            self.knot_slopes = jnp.concatenate(
                [ones, knot_slopes_core[..., 1:-1], ones], axis=-1
            )
        elif boundary_slopes == "circular":
            self.knot_slopes = jnp.concatenate(
                [knot_slopes_core[..., :-1], knot_slopes_core[..., :1]], axis=-1
            )
        else:
            raise ValueError(
                f"Unknown option for boundary slopes: `{boundary_slopes}`."
            )

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        fn = jnp.vectorize(
            _rational_quadratic_spline_fwd, signature="(),(n),(n),(n)->(),()"
        )
        y, logdet = fn(x, self.x_pos, self.y_pos, self.knot_slopes)
        return y, logdet

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        fn = jnp.vectorize(
            _rational_quadratic_spline_inv, signature="(),(n),(n),(n)->(),()"
        )
        x, logdet = fn(y, self.x_pos, self.y_pos, self.knot_slopes)
        return x, logdet

    def same_as(self, other: AbstractBijector) -> bool:
        return False
