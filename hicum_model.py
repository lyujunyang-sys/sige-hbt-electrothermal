# -*- coding: utf-8 -*-
"""
HICUM-style compact model for a SiGe HBT (reduced, temperature-aware).

This is the electrical half of the coupled system (Section 5.2 of the thesis).
Two modelling choices matter here:

1. Transfer current uses the **normalised base charge q_b** (integral charge-control,
   the actual HICUM/Gummel-Poon core), so the transfer current is counted once.

       I_T = I_S [exp(Vbe/(mcf Vt)) - exp(Vbc/(mcr Vt))] / q_b
       q_b = q1/2 + sqrt((q1/2)^2 + q2)
       q1  = 1 + Vbe/VAR + Vbc/VAF                      (Early effect)
       q2  = (I_S/IKF) exp(Vbe/(mcf Vt)) + (I_S/IKR) exp(Vbc/(mcr Vt))   (high injection)

2. All temperature-sensitive parameters are dimensionally-correct functions of the
   junction temperature T_j supplied by the thermal solver:
       I_S(T) = I_S0 (T/T0)^XTI exp[(Eg/kB)(1/T0 - 1/T)]
       R(T)   = R0 [1 + TC1 (T-T0) + TC2 (T-T0)^2]
       Vt     = kB T / q

Sign convention: NPN, emitter grounded. Internal junction voltages are the unknowns;
terminal currents are outputs.  Power dissipation P = Vce*Ic + Vbe*Ib  (-> heat source).
"""
import numpy as np

q = 1.602176634e-19      # C
kB = 1.380649e-23        # J/K


class HICUMParams:
    """Default SiGe-HBT-like parameter set (typical orders of magnitude)."""
    def __init__(self, **over):
        self.T0 = 300.0
        # transfer / saturation
        self.IS0 = 2.0e-16     # transfer saturation current (A)
        self.XTI = 3.0         # IS temperature exponent
        self.Eg = 1.10 * q     # bandgap (J)  (SiGe base, slightly < Si)
        self.mcf = 1.00        # forward emission coeff
        self.mcr = 1.00        # reverse emission coeff
        self.VAF = 60.0        # forward Early voltage (V)
        self.VAR = 8.0         # reverse Early voltage (V)
        self.IKF = 8.0e-3      # forward knee current (A)  -> high injection
        self.IKR = 4.0e-3      # reverse knee current (A)
        # base currents -> set DC beta and its bias/temperature shape
        self.IBES = 3.0e-18    # ideal B-E saturation current (A), ~n=1, ~ni^2
        self.IBEN = 4.0e-16    # NON-ideal B-E saturation current (A), n~2, ~ni
        self.IBCS = 1.0e-18    # B-C recombination saturation current (A)
        self.mBE = 1.05        # ideal B-E ideality factor
        self.mBEN = 2.0        # non-ideal (SCR recombination) ideality factor
        self.mBC = 1.10
        # series resistances (ohm) and their temperature coefficients
        self.RE0 = 3.0
        self.RC0 = 6.0
        self.RB0 = 40.0
        self.TC1 = 2.0e-3      # 1st-order temp coeff (1/K) for all R
        self.TC2 = 3.0e-6      # 2nd-order temp coeff (1/K^2)
        for k, v in over.items():
            setattr(self, k, v)


class HICUM:
    def __init__(self, params=None):
        self.p = params or HICUMParams()

    # ---- temperature-dependent parameters ------------------------------
    def temp_params(self, Tj):
        p = self.p
        T0 = p.T0
        dT = Tj - T0
        Vt = kB * Tj / q
        # ni^2-like activation (ideal currents) and ni-like activation (non-ideal)
        act2 = (Tj / T0) ** p.XTI * np.exp((p.Eg / kB) * (1.0 / T0 - 1.0 / Tj))
        act1 = (Tj / T0) ** (p.XTI / 2.0) * np.exp((p.Eg / (2.0 * kB)) * (1.0 / T0 - 1.0 / Tj))
        IS = p.IS0 * act2
        IBES = p.IBES * act2          # ideal base current ~ ni^2 (like IS)
        IBEN = p.IBEN * act1          # non-ideal (SCR) base current ~ ni
        rfac = 1.0 + p.TC1 * dT + p.TC2 * dT * dT
        return dict(Vt=Vt, IS=IS, IBES=IBES, IBEN=IBEN,
                    RE=p.RE0 * rfac, RC=p.RC0 * rfac, RB=p.RB0 * rfac)

    # ---- internal currents from *internal* junction voltages -----------
    def currents(self, Vbei, Vbci, Tj):
        """Return dict with Ic, Ib, Ie and the intermediate It, Ibe, Ibc."""
        p = self.p
        tp = self.temp_params(Tj)
        Vt, IS = tp["Vt"], tp["IS"]

        # clamp exponent arguments to avoid overflow during Newton excursions
        ef = np.clip(Vbei / (p.mcf * Vt), -40.0, 80.0)
        er = np.clip(Vbci / (p.mcr * Vt), -40.0, 80.0)
        Ef = np.exp(ef)
        Er = np.exp(er)

        # normalised base charge (integral charge control)
        q1 = 1.0 + max(Vbci, -p.VAF) / p.VAF + max(Vbei, -p.VAR) / p.VAR
        q2 = (IS / p.IKF) * Ef + (IS / p.IKR) * Er
        qb = 0.5 * q1 + np.sqrt((0.5 * q1) ** 2 + q2)

        It = IS * (Ef - Er) / qb                      # transfer current

        # base currents: ideal (n~1) + non-ideal SCR-recombination (n~2) components
        ebe = np.clip(Vbei / (p.mBE * Vt), -40.0, 80.0)
        eben = np.clip(Vbei / (p.mBEN * Vt), -40.0, 80.0)
        ebc = np.clip(Vbci / (p.mBC * Vt), -40.0, 80.0)
        Ibe = tp["IBES"] * (np.exp(ebe) - 1.0) + tp["IBEN"] * (np.exp(eben) - 1.0)
        Ibc = p.IBCS * (np.exp(ebc) - 1.0)

        Ic = It - Ibc
        Ib = Ibe + Ibc
        Ie = It + Ibe                                 # = Ic + Ib
        return dict(Ic=Ic, Ib=Ib, Ie=Ie, It=It, Ibe=Ibe, Ibc=Ibc, **tp)

    # ---- terminal solve: given terminal Vbe, Vce and Tj ----------------
    def solve_terminal(self, Vbe_term, Vce_term, Tj, x0=None,
                       max_iter=80, tol=1e-12):
        """
        Solve the 2x2 nonlinear system for internal (Vbei, Vbci) including the
        ohmic drops across RE, RC, RB:
            Vbe_term = Vbei + Ie*RE + Ib*RB
            Vbc_term = Vbci + Ib*RB - Ic*RC,   with Vbc_term = Vbe_term - Vce_term
        Returns (Vbei, Vbci, currents_dict).  Pure Newton with damping + FD Jacobian.
        """
        Vbc_term = Vbe_term - Vce_term
        if x0 is None:
            x = np.array([min(Vbe_term, 0.9), min(Vbc_term, 0.3)])
        else:
            x = np.array(x0, float)

        def residual(xv):
            Vbei, Vbci = xv
            c = self.currents(Vbei, Vbci, Tj)
            r1 = Vbei + c["Ie"] * c["RE"] + c["Ib"] * c["RB"] - Vbe_term
            r2 = Vbci + c["Ib"] * c["RB"] - c["Ic"] * c["RC"] - Vbc_term
            return np.array([r1, r2]), c

        for _ in range(max_iter):
            r, c = residual(x)
            if np.linalg.norm(r) < tol:
                break
            # finite-difference Jacobian (2x2)
            J = np.zeros((2, 2))
            for k in range(2):
                dxk = 1e-6 * max(1.0, abs(x[k]))
                xp = x.copy(); xp[k] += dxk
                rp, _ = residual(xp)
                J[:, k] = (rp - r) / dxk
            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dx = -r
            # damping + step limiting (junction voltages move slowly)
            dx = np.clip(dx, -0.1, 0.1)
            x = x + 0.8 * dx
        r, c = residual(x)
        return x[0], x[1], c

    def power(self, Vbe_term, Vce_term, c):
        """Dissipated power P = Vce*Ic + Vbe*Ib  (W)."""
        return Vce_term * c["Ic"] + Vbe_term * c["Ib"]


if __name__ == "__main__":
    h = HICUM()
    print("Isothermal Gummel @ Vce=2.0 V, T=300 K")
    print(" Vbe      Ic         Ib         beta")
    for Vbe in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        Vbei, Vbci, c = h.solve_terminal(Vbe, 2.0, 300.0)
        beta = c["Ic"] / c["Ib"] if c["Ib"] > 0 else float("nan")
        print(" %.2f   %.3e  %.3e   %6.1f" % (Vbe, c["Ic"], c["Ib"], beta))
    # temperature sensitivity check
    print("\nTemperature effect @ Vbe=0.8, Vce=2.0:")
    for Tj in [300.0, 350.0, 400.0]:
        _, _, c = h.solve_terminal(0.8, 2.0, Tj)
        print("  Tj=%.0f K  Ic=%.3e A  P=%.3e W" % (Tj, c["Ic"], h.power(0.8, 2.0, c)))
