"""Fit a triangular membership function to a Gaussian membership function.

A Gaussian has infinite support; a linguistic triangular term needs a finite
one. Given both peaks pinned to 1 and centred on the same mean, the only free
parameter is the triangle's half-width, expressed in units of the Gaussian's
``sigma``. This module picks that half-width by directly minimizing an error
integral between the two curves, rather than by an arbitrary rule of thumb
(e.g. a literal +/-3 sigma cutoff).

Setting ``g(x) = exp(-x**2/2)`` and ``t(x) = max(0, 1 - |x|/a)`` (both in units
of sigma, so this ratio is scale-free), the two constants below are the
half-width ``a`` that minimizes, respectively:

* ``GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH``: ``integral(|g(x) - t(x)| dx)`` (L1 / MAE)
* ``GAUSSIAN_TRIANGLE_MSE_HALF_WIDTH``: ``integral((g(x) - t(x))**2 dx)`` (L2 / MSE)

Both were found by golden-section search over Simpson-quadrature evaluations
of the respective integral (see the derivation in the PR/issue discussion for
#92); they are pinned by :mod:`tests.test_triangle_fit`. Widening the triangle
trades shoulder error against tail error and there is an interior optimum for
either objective -- a literal +/-3 sigma triangle is *worse* on both metrics
than either fitted width (roughly 2.5x the MAE-optimal error), because it
buys negligible extra tail coverage at the cost of a much worse match near
the shoulders, where a Gaussian is markedly more convex than a line.

Interestingly, ``GAUSSIAN_TRIANGLE_MSE_HALF_WIDTH`` (2.37547) matches, to four
decimal places, a constant (2.3756) that was already sitting -- unexplained
and unused -- in :func:`tribblefis.gauss_math.membership`'s triangular branch.
That code was apparently derived the same way for the MSE objective at some
point, but the derivation and the fact that it was MSE- rather than
MAE-optimal were never written down.
"""

from .gauss_data import GaussianMembership, GaussianMixtureModel, FeatureModel, LabelModel, TriangularMembership

#: MAE(L1)-optimal symmetric-triangle half-width, in units of sigma. Default
#: used by :func:`fit_triangle_to_gaussian` and by
#: :func:`tribblefis.ruspini.ruspinize_model`'s ``sigma_knots``.
GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH: float = 2.33293

#: MSE(L2)-optimal symmetric-triangle half-width, in units of sigma. Kept as a
#: named alternative for callers who specifically want the least-squares fit.
GAUSSIAN_TRIANGLE_MSE_HALF_WIDTH: float = 2.37547


def fit_triangle_to_gaussian(
    mf: GaussianMembership,
    half_width_sigma: float = GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
) -> TriangularMembership:
    """Fit a symmetric triangular membership to a Gaussian membership.

    The triangle is centred on ``mf.mu`` with apex 1 and half-width
    ``half_width_sigma * mf.sigma``. The Gaussian's own ``id`` is carried over
    onto the result, so any :class:`~tribblefis.gauss_data.Rule` that
    referenced ``mf`` by id keeps resolving correctly once it's replaced.
    """
    half_width = half_width_sigma * mf.sigma
    return TriangularMembership(
        a=mf.mu - half_width, b=mf.mu, c=mf.mu + half_width, id=mf.id
    )


def fit_triangles_to_mixture(
    model: GaussianMixtureModel,
    half_width_sigma: float = GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
) -> GaussianMixtureModel:
    """Return a copy of ``model`` with every Gaussian membership replaced by
    its fitted triangle (see :func:`fit_triangle_to_gaussian`).

    Non-Gaussian memberships (already-triangular or trapezoidal terms) pass
    through unchanged. Feature/label structure and membership ids are
    preserved, so this alone turns a Gaussian-based FIS into a triangle-based
    one without touching the rule base.
    """
    new_feature_models: dict[str, FeatureModel] = {}
    for feature_name, feature_model in model.feature_models.items():
        new_label_models: dict[int, LabelModel] = {}
        for label, label_model in feature_model.label_models.items():
            new_memberships = [
                fit_triangle_to_gaussian(mf, half_width_sigma)
                if isinstance(mf, GaussianMembership)
                else mf
                for mf in label_model.memberships
            ]
            new_label_models[label] = LabelModel(memberships=new_memberships)
        new_feature_models[feature_name] = FeatureModel(label_models=new_label_models)
    return GaussianMixtureModel(
        feature_models=new_feature_models, anomaly_params=model.anomaly_params
    )
