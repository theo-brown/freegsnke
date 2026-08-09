"""Plasma profiles specified as p(psi) and F(psi), differentiated by autodiff.

The Grad-Shafranov source is

    Jtor = R p'(psi) + FF'(psi) / (mu0 R)

Codes in this family normally ask the user for p' and FF' directly, and carry
optional p and F functions alongside so that pressure and toroidal field can be
reported (see ``freegs4e/jtor.py:1211-1231``, where ``pressure()`` falls back to
numerically integrating p'). Since we are already in JAX we invert that: the
user supplies the physical profiles p(psi) and F(psi), and the derivatives come
from ``jax.grad``:

    p'(psi)  = dp/dpsi
    FF'(psi) = F dF/dpsi = (1/2) d(F^2)/dpsi

so p' and p, and FF' and F, cannot disagree.

Profiles are functions of *unnormalised* psi with compact support: they are
constant below ``psi_edge``, so Jtor vanishes outside ``{psi > psi_edge}``
automatically. That is what removes the need to locate the last closed flux
surface during the solve -- see the module docstring of ``solver.py``.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import numpy as np
import jax.numpy as jnp

MU0 = 4e-7 * jnp.pi


class Profile(NamedTuple):
    """Pressure and poloidal-current functions of psi.

    Both are applied elementwise to an array of psi values.
    """

    p: Callable[[jnp.ndarray], jnp.ndarray]  # pressure [Pa]
    F: Callable[[jnp.ndarray], jnp.ndarray]  # F = R B_tor [T m]


def pprime(profile: Profile, psi):
    """dp/dpsi, elementwise.

    ``grad`` of the summed profile gives the elementwise derivative in a single
    pass, since p is applied pointwise.
    """
    return jax.grad(lambda q: jnp.sum(profile.p(q)))(psi)


def ffprime(profile: Profile, psi):
    """F dF/dpsi, elementwise, obtained as half the derivative of F^2.

    Differentiating F^2 rather than F avoids a separate division by F and stays
    well behaved as F approaches the vacuum value.
    """
    return 0.5 * jax.grad(lambda q: jnp.sum(profile.F(q) ** 2))(psi)


def jtor(profile: Profile, R, psi, vessel_mask, Ip=None, dA=None):
    """Toroidal current density on the grid.

    Parameters
    ----------
    R : (nR, nZ) major radius of each grid point.
    psi : (nR, nZ) total poloidal flux.
    vessel_mask : (nR, nZ) fixed geometric mask of the region where plasma is
        allowed. This is a compile-time constant, not a function of psi, so it
        is exactly differentiable (trivially: its derivative is zero).
    Ip : if given, the profile amplitude is rescaled so the total current equals
        ``Ip``. Jtor is linear in the profile amplitude, so this is a single
        division rather than an extra unknown -- the same device as
        ``Lao85``/``Fiesta_Topeol`` (``freegs4e/jtor.py:911-916``). Pass None to
        use the profiles as given.
    dA : cell area, required when ``Ip`` is given.
    """
    j = vessel_mask * (
        R * pprime(profile, psi) + ffprime(profile, psi) / (MU0 * R)
    )
    if Ip is None:
        return j
    total = jnp.sum(j) * dA
    return j * (Ip / total)


def lao85(psi_axis, psi_bndry, alpha, beta, L, fvac, Raxis):
    """Lao (1985) polynomial profiles, expressed as p(psi) and F(psi).

    FreeGS4E's ``Lao85`` specifies the derivatives on normalised flux
    (``freegs4e/jtor.py:868-889``)::

        Jtor = L [ (R/Raxis) sum_i alpha_i psiN^i
                 + (Raxis / (mu0 R)) sum_i beta_i psiN^i ]

    Matching term by term against ``Jtor = R p' + FF'/(mu0 R)`` gives
    ``p' = (L/Raxis) sum alpha_i psiN^i`` and
    ``FF' = L Raxis sum beta_i psiN^i``. Both integrate in closed form, so the
    physical profiles can be written directly and their derivatives recovered by
    autodiff -- reproducing Lao85 exactly rather than approximating it.

    Integration constants are fixed by p = 0 and F = fvac at the plasma edge.

    ``alpha`` and ``beta`` must be the *full* coefficient arrays, including the
    final term FreeGS4E appends when ``alpha_logic``/``beta_logic`` are set
    (``jtor.py:877-882``). That term forces ``sum_i alpha_i = 0``, i.e.
    ``p'(edge) = 0``, which is what gives the profile compact support. Note it
    is a *single* root: p'' does not vanish at the edge, so the Jacobian has a
    jump there. See ``compact`` for a family with a double root.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    D = psi_bndry - psi_axis  # negative: psi decreases outward from the axis

    def psi_norm(psi):
        n = (psi - psi_axis) / D
        # Clamp as FreeGS4E does with np.clip(psi_norm, 0, 1). The lower clamp
        # introduces a kink in the source at the axis flux; that is inherited
        # from the reference formulation, which can assume psi_axis really is
        # the maximum of psi, whereas here it is a prescribed parameter.
        return jnp.clip(n, 0.0, 1.0)

    def _antiderivative(n, coeffs):
        """sum_k c_k (n^(k+1) - 1) / (k+1), vanishing at n = 1.

        Evaluated by Horner with a Python-level loop so that every power is a
        structural multiplication. Writing it as ``n ** exps`` with an array of
        exponents routes through ``lax.pow``, whose second derivative
        ``y (y-1) x^(y-2)`` evaluates to ``0 * inf = NaN`` at ``x = 0, y = 1``.
        The value and first derivative are unaffected, so such a bug surfaces
        only once the Jacobian is taken -- exactly what Newton needs.
        """
        c = coeffs / np.arange(1, len(coeffs) + 1)  # coefficient of n^(k+1)
        acc = jnp.zeros_like(n)
        for ck in c[::-1]:
            acc = acc * n + ck
        return n * acc - float(c.sum())

    def p(psi):
        n = psi_norm(psi)
        return jnp.where(
            n < 1.0, (L * D / Raxis) * _antiderivative(n, alpha), 0.0
        )

    def F(psi):
        n = psi_norm(psi)
        excess = jnp.where(
            n < 1.0, 2.0 * L * Raxis * D * _antiderivative(n, beta), 0.0
        )
        return jnp.sqrt(fvac**2 + excess)

    return Profile(p, F)


def compact(psi_edge, psi_scale, p0, fvac, f1=0.0, alpha_p=2.5, alpha_f=2.5):
    """A simple profile family with compact support above ``psi_edge``.

    ``p = p0 * s**alpha_p`` and ``F = sqrt(fvac**2 + f1 * s**alpha_f)`` where
    ``s = max(0, (psi - psi_edge) / psi_scale)``.

    The exponents control smoothness at the plasma edge, and this matters for
    Newton. A zero of order ``alpha`` in p gives a zero of order ``alpha - 1``
    in p' and ``alpha - 2`` in p''. Since the Jacobian of the residual contains
    p'', ``alpha > 2`` is needed for the Jacobian to be continuous across the
    plasma edge and hence for clean quadratic convergence. The default 2.5
    satisfies this; note FreeGS's default Topeol shape corresponds to
    ``alpha_p = 3``.

    Parameters
    ----------
    psi_edge : flux at the plasma edge; the profile is flat below it.
    psi_scale : flux scale normalising the profile argument.
    p0 : pressure at ``s = 1``.
    fvac : vacuum field parameter R*Btor, the value of F outside the plasma.
    f1 : size of the diamagnetic/paramagnetic excursion of F^2 inside the plasma.
    """

    def _inside_and_safe(psi):
        """Return (inside, s_safe) where s_safe is never zero.

        The power must never be evaluated at zero even in a branch that is
        subsequently discarded: a NaN or infinity produced there still poisons
        the gradient. So substitute a dummy value of 1 outside the plasma and
        select afterwards.
        """
        u = (psi - psi_edge) / psi_scale
        inside = u > 0.0
        return inside, jnp.where(inside, u, 1.0)

    def p(psi):
        inside, s = _inside_and_safe(psi)
        return jnp.where(inside, p0 * s**alpha_p, 0.0)

    def F(psi):
        inside, s = _inside_and_safe(psi)
        f2 = jnp.where(inside, f1 * s**alpha_f, 0.0)
        return jnp.sqrt(fvac**2 + f2)

    return Profile(p, F)
