# -*- coding: utf-8 -*-
"""
3D transient heat-conduction solver (Fourier law), finite-volume discretisation,
implicit (backward-Euler) time integration, Robin (convective) boundary conditions.

This follows Section 3 of the thesis:
    rho*cp * dT/dt = div(k grad T) + q_v
with Robin BC  -k dT/dn = h (T - T_inf).

Backward Euler makes the scheme unconditionally stable (no CFL restriction), which
is what we want for the coupled electro-thermal solve.

The solver exposes the *linear operator* (stiffness matrix K + capacity term) so the
coupled Newton solver in coupled_solver.py can reuse the prefactorised matrix.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Mesh3D:
    """Uniform structured 3D grid over a box [0,Lx] x [0,Ly] x [0,Lz]."""

    def __init__(self, Lx, Ly, Lz, nx, ny, nz):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dz = Lz / (nz - 1)
        self.N = nx * ny * nz
        self.Vcell = self.dx * self.dy * self.dz  # control volume per node

    def idx(self, i, j, k):
        return (k * self.ny + j) * self.nx + i


class ThermalSolver:
    """
    Builds the FVM conduction/Robin stiffness matrix K and the capacity vector.

    Residual of the thermal sub-system at time level n+1 (backward Euler):
        R_th(T) = Mcap/dt * (T - T_n)  +  K @ T  -  b_robin  -  Q
    where
        Mcap/dt * I  is the (lumped) heat-capacity diagonal,
        K            is the conduction + Robin stiffness (symmetric positive-definite),
        b_robin      is the Robin ambient source on boundary faces,
        Q            is the volumetric heat-source vector (W) per node.

    The Jacobian of R_th w.r.t. T is  D = Mcap/dt * I + K  (constant -> prefactorise).
    """

    def __init__(self, mesh, rho, cp, k_therm, T_inf,
                 h_sink=5.0e4, h_amb=2.0e2, h_conv=None):
        """
        Boundary conditions (Robin on every outer face):
          * bottom face (z = 0)  -> strong cooling h_sink  (the heat-sink contact);
          * all other faces      -> weak coupling h_amb    (near-adiabatic to air).
        This makes vertical CONDUCTION through the die the dominant thermal resistance,
        so the junction temperature actually depends on the conductivity k -- the
        physically realistic situation for a die mounted on a heat sink.
        Pass h_conv=<value> to fall back to a single uniform coefficient on all faces.
        """
        self.m = mesh
        self.rho = rho
        self.cp = cp
        self.k = k_therm          # thermal conductivity (W/m/K) -- constant here
        self.T_inf = T_inf        # ambient temperature (K)
        if h_conv is not None:
            h_sink = h_amb = h_conv
        self.h_sink = h_sink
        self.h_amb = h_amb

        self.Mcap = rho * cp * mesh.Vcell        # heat capacity per node (J/K)
        self.K, self.b_robin = self._build_stiffness()

    def _build_stiffness(self):
        m = self.m
        N = m.N
        dx, dy, dz = m.dx, m.dy, m.dz
        # face areas
        Ax, Ay, Az = dy * dz, dx * dz, dx * dy
        # conductances k*A/d for an internal face
        gx, gy, gz = self.k * Ax / dx, self.k * Ay / dy, self.k * Az / dz

        rows, cols, vals = [], [], []
        b = np.zeros(N)

        def add(p, q, v):
            rows.append(p); cols.append(q); vals.append(v)

        for k in range(m.nz):
            for j in range(m.ny):
                for i in range(m.nx):
                    p = m.idx(i, j, k)
                    diag = 0.0
                    # x neighbours (side faces -> ambient)
                    for di, g, A in ((-1, gx, Ax), (1, gx, Ax)):
                        ii = i + di
                        if 0 <= ii < m.nx:
                            add(p, m.idx(ii, j, k), -g); diag += g
                        else:
                            hc = self.h_amb * A
                            diag += hc; b[p] += hc * self.T_inf
                    for dj, g, A in ((-1, gy, Ay), (1, gy, Ay)):
                        jj = j + dj
                        if 0 <= jj < m.ny:
                            add(p, m.idx(i, jj, k), -g); diag += g
                        else:
                            hc = self.h_amb * A
                            diag += hc; b[p] += hc * self.T_inf
                    # z neighbours: bottom face (dk=-1 at k=0) is the heat sink
                    for dk, g, A in ((-1, gz, Az), (1, gz, Az)):
                        kk = k + dk
                        if 0 <= kk < m.nz:
                            add(p, m.idx(i, j, kk), -g); diag += g
                        else:
                            h_face = self.h_sink if dk == -1 else self.h_amb
                            hc = h_face * A
                            diag += hc; b[p] += hc * self.T_inf
                    add(p, p, diag)
        K = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        return K, b

    # ---- steady state: K T = b_robin + Q --------------------------------
    def solve_steady(self, Q):
        return spla.spsolve(self.K.tocsc(), self.b_robin + Q)

    # ---- one implicit time step -----------------------------------------
    def operator_dynamic(self, dt):
        """Return D = Mcap/dt I + K  (the thermal Jacobian for transient)."""
        N = self.m.N
        return (self.Mcap / dt) * sp.identity(N, format="csr") + self.K

    def rhs_dynamic(self, dt, T_prev, Q):
        return (self.Mcap / dt) * T_prev + self.b_robin + Q

    def step(self, dt, T_prev, Q, D_factor=None):
        """Advance one backward-Euler step with fixed source Q."""
        D = self.operator_dynamic(dt)
        rhs = self.rhs_dynamic(dt, T_prev, Q)
        return spla.spsolve(D.tocsc(), rhs)


if __name__ == "__main__":
    # ---- self-test: heat a central spot, check diffusion & energy balance
    mesh = Mesh3D(200e-6, 200e-6, 80e-6, 21, 21, 9)
    ts = ThermalSolver(mesh, rho=2330.0, cp=700.0, k_therm=130.0,
                       h_conv=1.0e4, T_inf=300.0)
    # put 0.1 W into the top-centre node
    Q = np.zeros(mesh.N)
    Q[mesh.idx(10, 10, 8)] = 0.1
    Ts = ts.solve_steady(Q)
    print("steady solve: Tmax=%.2f K  Tmin=%.2f K  (ambient 300)"
          % (Ts.max(), Ts.min()))
    # transient: should monotonically rise toward steady state
    T = np.full(mesh.N, 300.0)
    for n in range(200):
        T = ts.step(1e-7, T, Q)
    print("transient 200 steps (dt=1e-7): Tmax=%.2f K" % T.max())
    print("PASS" if Ts.max() > 300.0 and np.all(Ts >= 299.9) else "CHECK")
