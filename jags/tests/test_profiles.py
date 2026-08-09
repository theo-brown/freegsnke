"""Profiles: autodiff derivatives, edge smoothness, and Lao85 equivalence."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags import profiles  # noqa: E402

MU0 = 4e-7 * np.pi


def central_diff(f, x, h):
    return (np.asarray(f(x + h)) - np.asarray(f(x - h))) / (2 * h)


def test_autodiff_derivatives_match_finite_differences():
    """p' and FF' come from jax.grad, so they cannot disagree with p and F."""
    pr = profiles.compact(0.4, 0.6, p0=1.2e4, fvac=0.5, f1=0.08)
    psi = jnp.linspace(0.45, 1.4, 25)

    np.testing.assert_allclose(
        np.asarray(profiles.pprime(pr, psi)),
        central_diff(pr.p, psi, 1e-7),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(profiles.ffprime(pr, psi)),
        0.5 * central_diff(lambda q: pr.F(q) ** 2, psi, 1e-7),
        rtol=1e-5,
    )

    # and both vanish identically below the plasma edge
    below = jnp.linspace(-0.5, 0.4, 10)
    assert np.abs(np.asarray(profiles.pprime(pr, below))).max() == 0.0
    assert np.abs(np.asarray(profiles.ffprime(pr, below))).max() == 0.0
    np.testing.assert_allclose(np.asarray(pr.F(below)), 0.5, rtol=1e-13)


@pytest.mark.parametrize("alpha_p", [2.5, 3.0, 4.0])
def test_second_derivative_continuous_at_edge(alpha_p):
    """p'' must tend to zero at the plasma edge for a continuous Jacobian.

    The residual Jacobian contains p'', so a jump there blunts Newton's
    quadratic convergence. A zero of order alpha in p leaves a zero of order
    alpha - 2 in p'', so alpha > 2 gives continuity -- but the approach can be
    slow: at alpha = 2.5, p'' only decays like sqrt(distance), so testing a
    fixed small offset against a fixed threshold would be measuring the decay
    rate, not continuity. Check the limit instead.
    """
    edge = 0.4
    pr = profiles.compact(edge, 0.6, p0=1.2e4, fvac=0.5, f1=0.08, alpha_p=alpha_p)
    d2 = jax.grad(lambda q: jnp.sum(profiles.pprime(pr, q)))

    assert float(d2(jnp.array([edge - 1e-9]))[0]) == 0.0

    offsets = np.array([1e-3, 1e-5, 1e-7, 1e-9])
    inside = np.array([float(d2(jnp.array([edge + o]))[0]) for o in offsets])
    assert np.all(np.diff(np.abs(inside)) < 0.0), "p'' should decay toward the edge"

    # The decay follows the predicted power law s**(alpha-2); a positive
    # exponent is exactly the statement that p'' tends to zero at the edge.
    rate = np.polyfit(np.log(offsets), np.log(np.abs(inside)), 1)[0]
    assert rate == pytest.approx(alpha_p - 2.0, abs=0.05)
    assert rate > 0.0


def test_lao85_second_derivative_is_finite():
    """Regression: the Jacobian of the Lao85 profile must not be NaN.

    Writing the polynomial as ``n ** exps`` with an *array* of exponents routes
    through ``lax.pow``, whose second derivative ``y (y-1) x^(y-2)`` evaluates to
    ``0 * inf = NaN`` at ``x = 0, y = 1``. Since psi_norm is clipped at 0, x = 0
    occurs wherever psi reaches the axis value. The value and first derivative
    are perfectly finite, so this only appears once the Jacobian is formed.
    """
    pr = profiles.lao85(
        psi_axis=0.0726, psi_bndry=0.0110,
        alpha=[2.0, -1.0, -1.0], beta=[1.0, -0.5, -0.5],
        L=0.61, fvac=0.5, Raxis=0.9,
    )
    # spans below the edge, inside, at the axis, and above it
    psi = jnp.array([-0.01, 0.011, 0.04, 0.0726, 0.0726 + 1e-9, 0.11])
    for fn in (profiles.pprime, profiles.ffprime):
        d2 = jax.jacfwd(lambda q: fn(pr, q))(psi)
        assert np.all(np.isfinite(np.asarray(d2))), f"{fn.__name__} Jacobian not finite"


def test_lao85_reproduces_reference_jtor_formula():
    """p and F integrate back to exactly the Lao85 current density.

    Compares against ``freegs4e/jtor.py:868-889`` written out directly:
        Jtor = L [ (R/Raxis) sum a_i psiN^i + (Raxis/(mu0 R)) sum b_i psiN^i ]
    """
    psi_axis, psi_bndry = 0.0726, 0.0110
    alpha = np.array([2.0, -1.0, -1.0])
    beta = np.array([1.0, -0.5, -0.5])
    L, fvac, Raxis = 0.61, 0.5, 0.9

    pr = profiles.lao85(psi_axis, psi_bndry, alpha, beta, L, fvac, Raxis)

    # Sampled strictly inside the plasma. The endpoints psi = psi_axis and
    # psi = psi_bndry are the two kinks of the clip, where the derivative is
    # one-sided; see test_lao85_endpoint_derivative_is_averaged.
    R = jnp.linspace(0.5, 1.5, 20)
    psi = jnp.linspace(psi_bndry, psi_axis, 22)[1:-1]
    n = np.clip((np.asarray(psi) - psi_axis) / (psi_bndry - psi_axis), 0.0, 1.0)

    poly_a = sum(alpha[i] * n**i for i in range(len(alpha)))
    poly_b = sum(beta[i] * n**i for i in range(len(beta)))
    want = L * (
        np.asarray(R) / Raxis * poly_a
        + Raxis / (MU0 * np.asarray(R)) * poly_b
    )

    got = np.asarray(R) * np.asarray(profiles.pprime(pr, psi)) + np.asarray(
        profiles.ffprime(pr, psi)
    ) / (MU0 * np.asarray(R))

    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_lao85_endpoint_derivative_is_averaged():
    """At exactly psi = psi_axis the clip kink halves p'.

    Documented rather than fixed. Deriving p' from p means differentiating
    through ``jnp.clip``, and JAX returns the averaged subgradient at the kink,
    whereas FreeGS4E evaluates its polynomial for p' directly and never sees
    one. The two therefore disagree by a factor of two on the single flux
    surface psi = psi_axis, and nowhere else. No grid point lands exactly there
    in practice -- in the cross-check the solved axis flux differs from the
    prescribed psi_axis in the sixth decimal -- so this does not affect the
    solution, but it would surface as a puzzling factor of two if a test ever
    sampled the endpoint.
    """
    psi_axis, psi_bndry = 0.0726, 0.0110
    alpha = np.array([2.0, -1.0, -1.0])
    pr = profiles.lao85(
        psi_axis, psi_bndry, alpha, [1.0, -0.5, -0.5], L=0.61, fvac=0.5, Raxis=0.9
    )
    at_axis = float(profiles.pprime(pr, jnp.array([psi_axis]))[0])
    just_inside = float(profiles.pprime(pr, jnp.array([psi_axis - 1e-9]))[0])
    assert at_axis == pytest.approx(0.5 * just_inside, rel=1e-6)


def test_jtor_respects_ip_constraint():
    from jags.grid import Grid

    box = np.array([[0.35, -0.85], [1.55, -0.85], [1.55, 0.85], [0.35, 0.85]])
    g = Grid(0.2, 2.0, -1.0, 1.0, 33, 33, box)
    pr = profiles.compact(0.0, 0.15, p0=8e3, fvac=0.5, f1=0.05)
    psi = jnp.asarray(np.exp(-(((g.R - 0.9) / 0.4) ** 2 + (g.Z / 0.6) ** 2)) * 0.3)
    mask = jnp.asarray(g.limiter_mask(), dtype=float)

    j = profiles.jtor(pr, jnp.asarray(g.R), psi, mask, Ip=4.2e5, dA=g.dA)
    assert float(jnp.sum(j)) * g.dA == pytest.approx(4.2e5, rel=1e-12)


def test_topeol_matches_the_freegs4e_current_expressions():
    """``topeol`` must reproduce FreeGS4E's ConstrainBetapIp derivatives exactly.

    FreeGS4E writes the current, not the profiles
    (``freegs4e/jtor.py:48-155``)::

        p'  = (L Beta0 / Raxis) (1 - psiN^m)^n
        FF' = mu0 L (1 - Beta0) Raxis (1 - psiN^m)^n

    jags integrates those to p(psi) and F(psi) and differentiates back with
    autodiff, so agreement here is a round trip through the antiderivative.
    This profile exists because ``lao85`` collapses at a large major radius;
    see its docstring.
    """
    pa, pb, L, B0, Ra, fvac, m, n = 2.0, 0.5, 1.234e6, 0.3, 6.2, 32.86, 2.0, 1
    prof = profiles.topeol(pa, pb, L, B0, Ra, fvac, alpha_m=m, alpha_n=n)

    # Interior only: at exactly psi_axis the clip kink halves the derivative,
    # the same documented behaviour as test_lao85_endpoint_derivative_is_averaged.
    psi = jnp.linspace(pb, pa, 9)[:-1]
    shape = (1.0 - ((pa - np.asarray(psi)) / (pa - pb)) ** m) ** n

    np.testing.assert_allclose(
        np.asarray(profiles.pprime(prof, psi)), L * B0 / Ra * shape, rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(profiles.ffprime(prof, psi)),
        profiles.MU0 * L * (1 - B0) * Ra * shape, rtol=1e-10,
    )
    # Compact support: nothing outside the plasma, and F back to vacuum.
    outside = jnp.asarray([pb - 1.0, pb - 0.1])
    np.testing.assert_allclose(np.asarray(prof.p(outside)), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(prof.F(outside)), fvac, rtol=1e-12)


def test_topeol_rejects_non_integer_alpha_n():
    """The antiderivative is a finite binomial sum only for integer alpha_n.

    For anything else it is an incomplete beta function, so the constructor
    refuses rather than quietly returning a profile that is not the one asked
    for.
    """
    with pytest.raises(ValueError, match="alpha_n"):
        profiles.topeol(2.0, 0.5, 1e6, 0.3, 6.2, 32.86, alpha_m=2.0, alpha_n=1.5)
