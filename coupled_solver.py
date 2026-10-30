# -*- coding: utf-8 -*-
"""
Electro-thermal COUPLED solver (Section 5.3 of the thesis).

Solves, simultaneously, the HICUM electrical system and the 3D transient
heat-diffusion PDE as a single nonlinear system, using Newton's method with a
Schur-complement reduction -- exactly the scheme described in the thesis:

    J = [ A  B ]      A = dRel/dx_el      (2x2 , electrical self-block)
        [ C  D ]      B = dRel/dT         (2xN , only the junction node column)
                      C = dRth/dx_el      (Nx2 , only the active-region rows)
                      D = dRth/dT         (NxN , conduction + capacity, ~SPD sparse)

Schur complement on the SMALL (electrical) block:
    S       = A - B D^-1 C                       (2x2)
    dx_el   = S^-1 ( -Rel + B D^-1 Rth )
    dT      = D^-1 ( -Rth - C dx_el )

Coupling physics:
  * the electrical block sees the junction temperature  Tj = T[node_j];
  * the thermal block sees the dissipated power  P = Vce*Ic + Vbe*Ib  injected as a
    volumetric source over the emitter active region.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from thermal_solver import Mesh3D, ThermalSolver
from hicum_model import HICUM


class CoupledDevice:
    def __init__(self, hicum: HICUM, thermal: ThermalSolver,
                 active_half=1):
        self.h = hicum
        self.t = thermal
        m = thermal.m
        # emitter active region: centred block on the TOP layer (k = nz-1)
        ic, jc, kc = m.nx // 2, m.ny // 2, m.nz - 1
        self.node_j = m.idx(ic, jc, kc)            # representative junction node
        act = []
        for di in range(-active_half, active_half + 1):
            for dj in range(-active_half, active_half + 1):
                act.append(m.idx(ic + di, jc + dj, kc))
        self.active = np.array(sorted(set(act)))
        self.n_active = len(self.active)

    # ---- assemble the heat-source vector from dissipated power ----------
    def heat_source(self, P):
        Q = np.zeros(self.t.m.N)
        Q[self.active] = P / self.n_active          # W per active node (real)
        return Q

    # ---- one coupled solve (steady or one transient step) ---------------
    def _solve(self, Vbe, Vce, Dt0, rhs_lin, x0, T0,
               max_iter=40, tol=1e-9):
        """
        Dt0     : clean thermal operator (K for steady; Mcap/dt I + K for transient)
        rhs_lin : linear part of thermal rhs (b_robin for steady;
                  Mcap/dt*T_prev + b_robin for transient)
        x0      : initial [Vbei, Vbci];  T0 : initial temperature field
        Returns (x_el, T, info)
        """
        m = self.t.m
        N = m.N
        x = np.array(x0, float)
        T = T0.copy()
        nj = self.node_j

        def elec_residual(xv, Tj):
            Vbei, Vbci = xv
            c = self.h.currents(Vbei, Vbci, Tj)
            r1 = Vbei + c["Ie"] * c["RE"] + c["Ib"] * c["RB"] - Vbe
            r2 = Vbci + c["Ib"] * c["RB"] - c["Ic"] * c["RC"] - (Vbe - Vce)
            return np.array([r1, r2]), c

        last_norm = np.inf
        for it in range(max_iter):
            Tj = T[nj]
            Rel, c = elec_residual(x, Tj)
            P = self.h.power(Vbe, Vce, c)
            Q = self.heat_source(P)
            Rth = Dt0 @ T - rhs_lin - Q

            norm = np.linalg.norm(Rel) + np.linalg.norm(Rth) / (self.t.Mcap + 1.0)
            if norm < tol:
                break

            # --- A = dRel/dx_el (2x2, FD) ---
            A = np.zeros((2, 2))
            for k in range(2):
                d = 1e-7 * max(1.0, abs(x[k]))
                xp = x.copy(); xp[k] += d
                rp, _ = elec_residual(xp, Tj)
                A[:, k] = (rp - Rel) / d
            # --- B = dRel/dTj (2-vector, FD) ---
            dT = 1e-4
            rB, _ = elec_residual(x, Tj + dT)
            Bj = (rB - Rel) / dT                      # 2-vector at column node_j
            # --- dP/dx_el and dP/dTj (FD) for C and the D correction ---
            dP_dx = np.zeros(2)
            for k in range(2):
                d = 1e-7 * max(1.0, abs(x[k]))
                xp = x.copy(); xp[k] += d
                _, cp = elec_residual(xp, Tj)
                dP_dx[k] = (self.h.power(Vbe, Vce, cp) - P) / d
            _, cT = elec_residual(x, Tj + dT)
            dP_dTj = (self.h.power(Vbe, Vce, cT) - P) / dT

            # --- C = dRth/dx_el (N x 2): only active rows = -dQ/dx_el ---
            Cdense = np.zeros((N, 2))
            Cdense[self.active, 0] = -dP_dx[0] / self.n_active
            Cdense[self.active, 1] = -dP_dx[1] / self.n_active
            # --- D = Dt0 - dQ/dT  (sparse rank update at column node_j) ---
            corr = sp.csc_matrix(
                (np.full(self.n_active, -dP_dTj / self.n_active),
                 (self.active, np.full(self.n_active, nj))), shape=(N, N))
            D = (Dt0 + corr).tocsc()

            # --- Schur complement on the 2x2 electrical block ---
            lu = spla.splu(D)
            W = np.column_stack([lu.solve(Cdense[:, 0]), lu.solve(Cdense[:, 1])])
            w = lu.solve(Rth)
            # B as 2xN with only column nj -> B@M = outer(Bj, M[nj])
            S = A - np.outer(Bj, W[nj, :])
            rhs = -Rel + Bj * w[nj]
            try:
                dx = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(S + 1e-12 * np.eye(2), rhs, rcond=None)[0]
            dTfield = -w - W @ dx

            # damping + electrical step limit
            dx = np.clip(dx, -0.1, 0.1)
            alpha = 1.0 if it > 2 else 0.6
            x = x + alpha * dx
            T = np.maximum(T + alpha * dTfield, self.t.T_inf - 5.0)  # physical floor
            last_norm = norm

        Tj = T[nj]
        _, c = elec_residual(x, Tj)
        P = self.h.power(Vbe, Vce, c)
        return x, T, dict(iters=it, norm=last_norm, c=c, P=P, Tj=Tj)

    # ---- public: steady-state operating point ---------------------------
    def steady(self, Vbe, Vce, x0=None, T0=None):
        if x0 is None:
            x0 = [min(Vbe, 0.9), min(Vbe - Vce, 0.3)]
        if T0 is None:
            T0 = np.full(self.t.m.N, self.t.T_inf)
        return self._solve(Vbe, Vce, self.t.K.tocsc(), self.t.b_robin, x0, T0)

    # ---- public: transient self-heating under a bias step ---------------
    def transient(self, Vbe, Vce, dt, n_steps, T_start=None):
        m = self.t.m
        if T_start is None:
            T_start = np.full(m.N, self.t.T_inf)
        Dt0 = self.t.operator_dynamic(dt).tocsc()
        T = T_start.copy()
        x = np.array([min(Vbe, 0.9), min(Vbe - Vce, 0.3)])
        ts, Tj_hist, Ic_hist = [], [], []
        for n in range(n_steps):
            rhs_lin = (self.t.Mcap / dt) * T + self.t.b_robin
            x, T, info = self._solve(Vbe, Vce, Dt0, rhs_lin, x, T)
            ts.append((n + 1) * dt)
            Tj_hist.append(info["Tj"])
            Ic_hist.append(info["c"]["Ic"])
        return (np.array(ts), np.array(Tj_hist), np.array(Ic_hist), T)


def default_device(nx=15, ny=15, nz=7, params=None):
    mesh = Mesh3D(200e-6, 200e-6, 80e-6, nx, ny, nz)
    thermal = ThermalSolver(mesh, rho=2330.0, cp=700.0, k_therm=130.0,
                            h_conv=3.0e4, T_inf=300.0)
    return CoupledDevice(HICUM(params), thermal, active_half=1)


if __name__ == "__main__":
    dev = default_device()
    print("Active nodes: %d   junction node idx: %d" % (dev.n_active, dev.node_j))

    print("\n--- Steady operating points (with self-heating) ---")
    print(" Vbe   Vce    Ic         P(W)      Tj(K)   newton_it")
    x0, T0 = None, None
    for Vbe in [0.7, 0.75, 0.8, 0.85, 0.9]:
        x, T, info = dev.steady(Vbe, 2.0, x0, T0)
        x0, T0 = x, T
        print(" %.2f  %.1f   %.3e  %.3e  %.2f   %d"
              % (Vbe, 2.0, info["c"]["Ic"], info["P"], info["Tj"], info["iters"]))

    print("\n--- Transient self-heating @ Vbe=0.85, Vce=2.0 ---")
    ts, Tj, Ic, Tfield = dev.transient(0.85, 2.0, dt=2e-7, n_steps=80)
    print(" t=%.2e s  Tj_start=%.2f  Tj_end=%.2f K  Ic_start=%.3e  Ic_end=%.3e"
          % (ts[-1], Tj[0], Tj[-1], Ic[0], Ic[-1]))
    print(" Tj monotonic rising:", bool(np.all(np.diff(Tj) >= -1e-6)))
