# -*- coding: utf-8 -*-
"""
TWIN EXPERIMENT  (self-contained validation of the calibration workflow).

Without measured data for this device, the calibration is validated by
*parameter recovery*:

  1. choose a known "true" parameter vector theta_true;
  2. run the forward coupled electro-thermal model to produce synthetic "measurements"
     (a DC Gummel I_C(Vbe) curve + a transient self-heating Tj(t) curve) and ADD NOISE
     so they look like real measurements;
  3. forget theta_true and let Bayesian optimization minimise
            L(theta) = w_I * mean[(log Ic_sim - log Ic_meas)^2]
                     + w_T * mean[((Tj_sim - Tj_meas)/Tscale)^2]
     (this is the thesis Eq. in Section 5.1, ||y_sim(theta) - y_exp||^2 form);
  4. check that BO recovers theta_true and that L converges.

If BO recovers the hidden parameters from noisy data, the *method* (coupled solver +
GP/EI Bayesian calibration) is demonstrably working.  All data here is SYNTHETIC
and labelled as such.
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_solver import Mesh3D, ThermalSolver
from hicum_model import HICUM, HICUMParams
from coupled_solver import CoupledDevice
from bayes_opt import Param, Space, BayesianOptimizer

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2024)

# ---------------- device + experiment configuration ----------------------
NX, NY, NZ = 15, 15, 7
H_SINK = 5.0e4                 # bottom heat-sink contact (W/m^2/K)
H_AMB = 2.0e2                  # near-adiabatic on the other faces
VBE_GUMMEL = np.array([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
VCE = 2.0
# low-bias Gummel at an elevated chuck temperature: negligible self-heating, so it
# isolates the intrinsic IS(T) temperature dependence (XTI) from self-heating (k)
# low-bias Gummel at several chuck temperatures: the TEMPERATURE SLOPE of the current
# pins XTI, while the 300 K level pins IS0 -- this breaks the IS0<->XTI correlation.
T_AMB_LIST = [330.0, 360.0]
VBE_LOWT = np.array([0.55, 0.60, 0.65])
VBE_TR, VCE_TR = 0.90, 2.0     # transient self-heating bias
# cover several thermal time constants (tau ~ Rth*Cth ~ a few ms) so the steady
# self-heating level -- which is what fixes the thermal resistance / k -- is reached
DT_TR, NSTEP_TR = 3e-4, 40
T_SAMPLES = np.array([2, 5, 9, 15, 24, 39])   # transient steps sampled as "measured"
TJ_NOISE = 0.3                              # K, on-chip thermal-diode accuracy

# parameters being calibrated (4-D, well within BO's comfortable < 20-D regime).
# All four are genuinely identifiable from a Gummel curve + a transient Tj(t) curve:
#   IS0 sets the current level; XTI sets how strongly self-heating boosts Ic;
#   RE0 sets the high-injection roll-off; k sets the thermal resistance / Tj rise.
#   name        true value     search bounds            log?
CAL = [
    ("IS0",  2.0e-16, 5.0e-17, 8.0e-16, True),
    ("XTI",  3.0,     2.0,     4.5,     False),   # IS temperature exponent
    ("RE0",  3.0,     1.0,     8.0,     False),
    ("k",    130.0,   60.0,    220.0,   False),   # thermal conductivity (W/m/K)
]
THETA_TRUE = np.array([c[1] for c in CAL])
SPACE = Space([Param(n, lo, hi, log) for (n, _, lo, hi, log) in CAL])


def build_device(theta, T_inf=300.0):
    IS0, XTI, RE0, k = theta
    params = HICUMParams(IS0=IS0, XTI=XTI, RE0=RE0)
    mesh = Mesh3D(200e-6, 200e-6, 80e-6, NX, NY, NZ)
    # bottom = heat sink (strong), other faces near-adiabatic -> conduction (k) sets Rth
    thermal = ThermalSolver(mesh, rho=2330.0, cp=700.0, k_therm=k,
                            T_inf=T_inf, h_sink=H_SINK, h_amb=H_AMB)
    return CoupledDevice(HICUM(params), thermal, active_half=1)


def _gummel(dev, vbe_list):
    Ic = np.zeros(len(vbe_list))
    x0, T0 = None, None
    for i, vbe in enumerate(vbe_list):
        x0, T0, info = dev.steady(vbe, VCE, x0, T0)
        Ic[i] = info["c"]["Ic"]
    return Ic


def forward(theta):
    """Return (Ic@300K[6], Tj_transient[NSTEP], ts, [Ic_lowbias@T for T in T_AMB_LIST])."""
    dev = build_device(theta, T_inf=300.0)
    Ic = _gummel(dev, VBE_GUMMEL)
    ts, Tj, _, _ = dev.transient(VBE_TR, VCE_TR, DT_TR, NSTEP_TR)
    Ic_hot = [_gummel(build_device(theta, T_inf=T), VBE_LOWT) for T in T_AMB_LIST]
    return Ic, Tj, ts, Ic_hot


# ---------------- 1) synthetic "measurements" ----------------------------
print("=" * 64)
print(" TWIN EXPERIMENT  (data is SYNTHETIC, generated from theta_true)")
print("=" * 64)
Ic_true, Tj_true, ts_tr, Ichot_true = forward(THETA_TRUE)
Ic_meas = Ic_true * (1.0 + 0.03 * RNG.standard_normal(Ic_true.shape))      # 3% noise
Tj_meas = Tj_true + TJ_NOISE * RNG.standard_normal(Tj_true.shape)          # K noise
Ichot_meas = [c * (1.0 + 0.03 * RNG.standard_normal(c.shape)) for c in Ichot_true]
print("theta_true       :", THETA_TRUE)
print("Gummel Ic@300K   :", np.array2string(Ic_meas, precision=3))
for T, c in zip(T_AMB_LIST, Ichot_meas):
    print("Gummel Ic@%-5.0fK :" % T, np.array2string(c, precision=3), "(low-bias)")
print("Tj transient     : start=%.2f K  end=%.2f K  (rise %.2f K)"
      % (Tj_meas[0], Tj_meas[-1], Tj_meas[-1] - Tj_meas[0]))

T_SCALE = max(1.0, Tj_meas[-1] - Tj_meas[0])
logIc_meas = np.log(Ic_meas)
logIchot_meas = [np.log(c) for c in Ichot_meas]


# ---------------- 2) objective (thesis Section 5.1) ----------------------
def objective(theta):
    try:
        Ic, Tj, _, Ic_hot = forward(theta)
        if np.any(Ic <= 0) or np.any(~np.isfinite(Ic)) or \
           any(np.any(c <= 0) or np.any(~np.isfinite(c)) for c in Ic_hot):
            return 50.0
        e_I = np.mean((np.log(Ic) - logIc_meas) ** 2)
        e_T = np.mean(((Tj[T_SAMPLES] - Tj_meas[T_SAMPLES]) / T_SCALE) ** 2)
        e_H = np.mean([np.mean((np.log(c) - lm) ** 2)             # multi-T -> XTI
                       for c, lm in zip(Ic_hot, logIchot_meas)])
        return 1.0 * e_I + 1.5 * e_T + 1.5 * e_H
    except Exception:
        return 50.0


# ---------------- 3) Bayesian calibration --------------------------------
print("\nRunning Bayesian optimization (GP + EI)...")
bo = BayesianOptimizer(SPACE, xi=0.01, seed=7)
best_theta, best_L = bo.run(objective, n_init=16, n_iter=70, verbose=True)

print("\n" + "-" * 64)
print("RESULT")
print("-" * 64)
print(" param     true          recovered     rel.err")
for (name, true, lo, hi, log), rec in zip(CAL, best_theta):
    print("  %-5s   %-12.4g  %-12.4g  %6.1f%%"
          % (name, true, rec, 100 * abs(rec - true) / true))
print(" best objective L = %.4e" % best_L)


# ---------------- 4) figures ---------------------------------------------
Ic_fit, Tj_fit, ts_fit, _ = forward(best_theta)
Ic_init, Tj_init, _, _ = forward(SPACE.from_unit(np.full(SPACE.d, 0.5)))  # mid-box guess

fig = plt.figure(figsize=(13, 9))

ax1 = fig.add_subplot(2, 2, 1)
ax1.semilogy(VBE_GUMMEL, Ic_meas, "ko", ms=7, label="synthetic measurement")
ax1.semilogy(VBE_GUMMEL, Ic_fit, "r-", lw=2, label="BO-calibrated")
ax1.semilogy(VBE_GUMMEL, Ic_init, "b--", lw=1.5, label="initial guess (box centre)")
ax1.set_xlabel("V_BE (V)"); ax1.set_ylabel("I_C (A)")
ax1.set_title("Gummel I_C(V_BE) @ V_CE=2 V (with self-heating)")
ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(ts_tr * 1e6, Tj_meas, "ko", ms=4, label="synthetic measurement")
ax2.plot(ts_fit * 1e6, Tj_fit, "r-", lw=2, label="BO-calibrated")
ax2.plot(ts_fit * 1e6, Tj_init, "b--", lw=1.5, label="initial guess")
ax2.set_xlabel("time (us)"); ax2.set_ylabel("junction T_j (K)")
ax2.set_title("Transient self-heating @ V_BE=0.90 V")
ax2.legend(); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(bo.history_best(), "o-", ms=3)
ax3.set_xlabel("evaluation"); ax3.set_ylabel("best objective L")
ax3.set_yscale("log"); ax3.set_title("BO convergence"); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(2, 2, 4)
names = [c[0] for c in CAL]
xpos = np.arange(len(names))
ax4.bar(xpos - 0.2, THETA_TRUE / THETA_TRUE, 0.4, label="true (normalised=1)")
ax4.bar(xpos + 0.2, best_theta / THETA_TRUE, 0.4, label="recovered / true")
ax4.set_xticks(xpos); ax4.set_xticklabels(names)
ax4.axhline(1.0, color="k", lw=0.8)
ax4.set_ylabel("value / true value")
ax4.set_title("Parameter recovery"); ax4.legend(); ax4.grid(True, axis="y", alpha=0.3)

fig.suptitle("SiGe HBT electro-thermal coupling + Bayesian calibration "
             "(twin experiment, synthetic data)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_calibration.png")
fig.savefig(out, dpi=120)
print("\nSaved figure -> %s" % out)

# ---- temperature-field snapshot at the hottest operating point ----------
dev = build_device(best_theta)
x0, T0 = None, None
for vbe in VBE_GUMMEL:                      # warm-start sweep up to 0.90 V
    x0, T0, info = dev.steady(vbe, 2.0, x0, T0)
Tfield = T0
m = dev.t.m
Tvol = Tfield.reshape(m.nz, m.ny, m.nx)
fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))
im0 = axes[0].imshow(Tvol[-1], origin="lower", cmap="inferno",
                     extent=[0, m.Lx * 1e6, 0, m.Ly * 1e6])
axes[0].set_title("Top-surface T (K), V_BE=0.90 V, P=%.1f mW" % (info["P"] * 1e3))
axes[0].set_xlabel("x (um)"); axes[0].set_ylabel("y (um)")
fig2.colorbar(im0, ax=axes[0])
midj = m.ny // 2
im1 = axes[1].imshow(Tvol[:, midj, :], origin="lower", cmap="inferno", aspect="auto",
                     extent=[0, m.Lx * 1e6, 0, m.Lz * 1e6])
axes[1].set_title("Vertical slice T (K) through emitter")
axes[1].set_xlabel("x (um)"); axes[1].set_ylabel("z (um)")
fig2.colorbar(im1, ax=axes[1])
fig2.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_thermal_field.png")
fig2.savefig(out2, dpi=120)
print("Saved figure -> %s" % out2)
print("\nDone.")
