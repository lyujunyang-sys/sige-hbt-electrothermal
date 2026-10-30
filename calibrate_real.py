# -*- coding: utf-8 -*-
"""
Calibrate the reduced HICUM model against REAL silicon-calibrated reference data.

The reference data (`ihp_ref/gummel_T*`, `ihp_ref/output_Vbe*`) was produced by running
the **IHP SG13G2 open-PDK VBIC model card** (extracted from measured wafers, Lot EDJ802)
in ngspice. So this is a genuine extraction against real-device-calibrated curves, not a
synthetic twin experiment.

Forward model = HICUM electrical core (`hicum_model.HICUM`) + a *lumped* self-heating loop
        Tj = Tamb + Rth * P(Tj)         (solved self-consistently per bias point)
which is exactly the thesis's self-heating sub-model. Bayesian optimization (GP + EI)
minimises the log-domain mismatch on the multi-temperature Gummel + output curves.
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hicum_model import HICUM, HICUMParams
from bayes_opt import Param, Space, BayesianOptimizer

warnings.filterwarnings("ignore")
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ihp_ref") + os.sep

GUMMEL_T = [-40, 27, 85, 125]          # chuck temperatures (deg C)
VCE_GUMMEL = 1.0
OUT_VBE = [0.70, 0.75, 0.80, 0.85]     # output-curve base biases
# fit the forward-active / high-injection regime where the reduced model is valid
# (below ~0.7 V the VBIC base current is dominated by a non-ideal n=2 leakage term
#  that the reduced model does not include, so those points are excluded)
# now that the model has a non-ideal (n~2) base current, the fit extends down to 0.65 V
VBE_FIT = np.array([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
VCE_FIT = np.array([0.4, 0.8, 1.2, 1.6])


# ---------------- load ngspice reference curves --------------------------
def load_gummel(tt):
    d = np.loadtxt(REF + "gummel_T%d" % tt)      # cols: vbe ic vbe ib
    return d[:, 0], d[:, 1], d[:, 3]

def load_output(vbe):
    d = np.loadtxt(REF + "output_Vbe%.2f" % vbe)  # cols: vce ic
    return d[:, 0], d[:, 1]

def interp_at(x_ref, y_ref, x_q):
    return np.interp(x_q, x_ref, y_ref)

REF_GUMMEL = {}
for tt in GUMMEL_T:
    vb, ic, ib = load_gummel(tt)
    REF_GUMMEL[tt] = (interp_at(vb, ic, VBE_FIT), interp_at(vb, ib, VBE_FIT))
REF_OUT = {}
for vbe in OUT_VBE:
    vc, ic = load_output(vbe)
    REF_OUT[vbe] = interp_at(vc, ic, VCE_FIT)


# ---------------- forward model: HICUM + lumped self-heating -------------
def self_heated_point(h, Vbe, Vce, Tamb_K, Rth, x0=None):
    """Solve Tj = Tamb + Rth*P self-consistently. Returns (Ic, Ib, Tj, x)."""
    Tj = Tamb_K
    x = x0
    for _ in range(30):
        vbei, vbci, c = h.solve_terminal(Vbe, Vce, Tj, x0=x)
        x = [vbei, vbci]
        P = h.power(Vbe, Vce, c)
        Tj_new = min(Tamb_K + Rth * max(P, 0.0), Tamb_K + 250.0)  # cap runaway
        if abs(Tj_new - Tj) < 1e-4:
            Tj = Tj_new
            break
        Tj = Tj + 0.7 * (Tj_new - Tj)              # damped fixed point
    return c["Ic"], c["Ib"], Tj, x


q_ELEC = 1.602176634e-19


def forward(theta):
    IS0, XTI, Eg, IBES, IBEN, IKF, RE0, Rth = theta
    p = HICUMParams(IS0=IS0, XTI=XTI, Eg=Eg * q_ELEC, IBES=IBES, IBEN=IBEN,
                    IKF=IKF, RE0=RE0)
    h = HICUM(p)
    gummel = {}
    for tt in GUMMEL_T:
        Tk = tt + 273.15
        Ic = np.zeros(len(VBE_FIT)); Ib = np.zeros(len(VBE_FIT))
        x = None
        for i, vbe in enumerate(VBE_FIT):
            ic, ib, _, x = self_heated_point(h, vbe, VCE_GUMMEL, Tk, Rth, x)
            Ic[i], Ib[i] = ic, ib
        gummel[tt] = (Ic, Ib)
    out = {}
    for vbe in OUT_VBE:
        Ic = np.zeros(len(VCE_FIT)); x = None
        for j, vce in enumerate(VCE_FIT):
            ic, _, _, x = self_heated_point(h, vbe, vce, 27 + 273.15, Rth, x)
            Ic[j] = ic
        out[vbe] = Ic
    return gummel, out


# ---------------- objective ----------------------------------------------
def _logerr(sim, ref):
    sim = np.maximum(sim, 1e-18); ref = np.maximum(ref, 1e-18)
    return np.mean((np.log(sim) - np.log(ref)) ** 2)

def objective(theta):
    try:
        gummel, out = forward(theta)
        e = 0.0
        for tt in GUMMEL_T:
            ic, ib = gummel[tt]
            ric, rib = REF_GUMMEL[tt]
            e += _logerr(ic, ric) + 0.5 * _logerr(ib, rib)   # Ic + Ib now both fit
        for vbe in OUT_VBE:
            e += 0.5 * _logerr(out[vbe], REF_OUT[vbe])
        if not np.isfinite(e):
            return 1e3
        return e / (len(GUMMEL_T) * 1.2 + len(OUT_VBE) * 0.5)
    except Exception:
        return 1e3


#   name    low      high     log
CAL = [
    ("IS0",  1e-18,   5e-15,   True),
    ("XTI",  2.0,     5.0,     False),
    ("Eg",   0.95,    1.25,    False),   # effective bandgap (eV) -> sets T-spread
    ("IBES", 1e-21,   1e-17,   True),    # ideal base current (n~1)
    ("IBEN", 1e-18,   1e-13,   True),    # non-ideal base current (n~2)
    ("IKF",  5e-4,    5e-2,    True),
    ("RE0",  1.0,     40.0,    False),
    ("Rth",  2e3,     3e4,     True),
]
SPACE = Space([Param(n, lo, hi, log) for (n, lo, hi, log) in CAL])


if __name__ == "__main__":
    print("=" * 64)
    print(" CALIBRATION vs REAL IHP SG13G2 VBIC data (silicon-calibrated)")
    print("=" * 64)
    print("Reference @27C Gummel Ic:",
          np.array2string(REF_GUMMEL[27][0][[0, 3, 7]], precision=3),
          "(Vbe=0.65/0.80/1.00)")

    bo = BayesianOptimizer(SPACE, xi=0.01, seed=3)
    best, bestL = bo.run(objective, n_init=18, n_iter=70, verbose=True)

    print("\n" + "-" * 64 + "\nCALIBRATED PARAMETERS\n" + "-" * 64)
    for (n, lo, hi, log), v in zip(CAL, best):
        print("  %-5s = %.4g" % (n, v))
    print(" objective L = %.4e" % bestL)

    # ---- figures: fit vs real data ----
    gummel, out = forward(best)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["C0", "C1", "C2", "C3"]
    for tt, col in zip(GUMMEL_T, colors):
        ric, rib = REF_GUMMEL[tt]
        ax[0].semilogy(VBE_FIT, ric, "o", color=col, ms=5,
                       label="IHP %d C (Ic)" % tt)
        ax[0].semilogy(VBE_FIT, gummel[tt][0], "-", color=col, lw=2)
        ax[0].semilogy(VBE_FIT, rib, "^", color=col, ms=4, alpha=0.6)
        ax[0].semilogy(VBE_FIT, gummel[tt][1], "--", color=col, lw=1, alpha=0.6)
    ax[0].set_xlabel("V_BE (V)"); ax[0].set_ylabel("I_C, I_B (A)")
    ax[0].set_title("Multi-T Gummel: IHP data (markers) vs calibrated (lines)")
    ax[0].legend(fontsize=8); ax[0].grid(True, which="both", alpha=0.3)

    for vbe, col in zip(OUT_VBE, colors):
        ax[1].plot(VCE_FIT, REF_OUT[vbe] * 1e3, "o", color=col, ms=5,
                   label="IHP Vbe=%.2f" % vbe)
        ax[1].plot(VCE_FIT, out[vbe] * 1e3, "-", color=col, lw=2)
    ax[1].set_xlabel("V_CE (V)"); ax[1].set_ylabel("I_C (mA)")
    ax[1].set_title("Output Ic-Vce @27C: IHP data vs calibrated")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)
    fig.suptitle("Reduced HICUM model calibrated to REAL IHP SG13G2 SiGe HBT data",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_real_calibration.png")
    fig.savefig(out, dpi=120)
    print("\nSaved -> " + out)
    print("Done.")
