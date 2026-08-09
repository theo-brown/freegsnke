"""Extract the ITER hybrid shape target from TORAX's own equilibrium file.

``iterhybrid_cocos11.eqdsk`` is the CHEASE equilibrium behind TORAX's
``iterhybrid_rampup`` and ``iterhybrid_predictor_corrector`` configs. Everything
needed to pose an inverse solve for the same plasma is in it: the last closed
flux surface, the plasma current, and ``F = R B_phi`` in the vacuum.

Writes a small .npz because the FreeGSNKE venv (numpy < 2, python 3.10) has no
``eqdsk`` package -- the same two-venv split as the rest of the cross-checks.

Usage:
    /home/user/torax/.venv/bin/python scripts/iter_target.py [out.npz]
"""

import sys

import numpy as np

EQDSK = "/home/user/torax/torax/data/third_party/geo/iterhybrid_cocos11.eqdsk"
N_ISOFLUX = 24


def main(out_path="scripts/iter_target.npz", gfile=EQDSK):
    import eqdsk

    d = eqdsk.EQDSKInterface.from_file(gfile, from_cocos=11, no_cocos=False).__dict__
    Rb, Zb = np.asarray(d["xbdry"]), np.asarray(d["zbdry"])
    R_axis, Z_axis = float(d["xmag"]), float(d["zmag"])

    # The X-point is the bottom of the separatrix: this is a lower single null.
    ix = int(np.argmin(Zb))
    Rx, Zx = float(Rb[ix]), float(Zb[ix])

    # Isoflux points, spread evenly in geometric poloidal angle about the axis
    # so the whole boundary is constrained rather than wherever the file happens
    # to have put its points. The X-point is added separately as a null.
    theta = np.arctan2(Zb - Z_axis, Rb - R_axis)
    order = np.argsort(theta)
    pick = order[np.linspace(0, len(order) - 1, N_ISOFLUX, dtype=int)]
    iso_R, iso_Z = Rb[pick], Zb[pick]

    np.savez_compressed(
        out_path,
        lcfs_R=Rb, lcfs_Z=Zb,
        isoflux_R=iso_R, isoflux_Z=iso_Z,
        Rx=Rx, Zx=Zx, R_axis=R_axis, Z_axis=Z_axis,
        Ip=abs(float(d["cplasma"])),
        fvac=float(d["bcentre"] * d["xcentre"]),
        B_0=float(d["bcentre"]), R_0=float(d["xcentre"]),
        p_axis=float(np.asarray(d["pressure"])[0]),
        R_geo=float((Rb.max() + Rb.min()) / 2),
        a_minor=float((Rb.max() - Rb.min()) / 2),
    )
    print(f"wrote {out_path}")
    print(f"  Ip        = {abs(float(d['cplasma'])):.4e} A")
    print(f"  fvac      = {float(d['bcentre'] * d['xcentre']):.4f} m T")
    print(f"  axis      = ({R_axis:.4f}, {Z_axis:.4f})")
    print(f"  X-point   = ({Rx:.4f}, {Zx:.4f})")
    print(f"  R_geo, a  = {(Rb.max() + Rb.min()) / 2:.4f}, "
          f"{(Rb.max() - Rb.min()) / 2:.4f}")
    print(f"  p_axis    = {float(np.asarray(d['pressure'])[0]):.4e} Pa")
    print(f"  {N_ISOFLUX} isoflux points, R {iso_R.min():.3f}-{iso_R.max():.3f}, "
          f"Z {iso_Z.min():.3f}-{iso_Z.max():.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scripts/iter_target.npz")
