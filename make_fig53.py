# -*- coding: utf-8 -*-
"""Generate the 4-panel Section 5.3 figure from the coupled electro-thermal solver."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from thermal_solver import Mesh3D, ThermalSolver
from hicum_model import HICUM
from coupled_solver import CoupledDevice

mesh = Mesh3D(200e-6, 200e-6, 80e-6, 21, 21, 9)
thermal = ThermalSolver(mesh, rho=2330.0, cp=700.0, k_therm=130.0,
                        T_inf=300.0, h_sink=4.0e3, h_amb=4.0e3)  # modest cooling
dev = CoupledDevice(HICUM(), thermal, active_half=1)

Vbe, Vce = 0.9, 2.0
# cover several thermal time constants (tau = Rth*Cth ~ a few ms) to reach steady self-heating
ts, Tj, Ic, Tfield = dev.transient(Vbe, Vce, dt=3e-4, n_steps=60)
print("Ic: %.3f -> %.3f mA   Tj: %.1f -> %.1f K"
      % (Ic[0]*1e3, Ic[-1]*1e3, Tj[0], Tj[-1]))

fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(ts*1e6, Ic*1e3, "b-"); ax1.set_xlabel("time (us)")
ax1.set_ylabel("I_C (mA)"); ax1.set_title("Collector current vs. time")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(ts*1e6, Tj, "r-"); ax2.set_xlabel("time (us)")
ax2.set_ylabel("junction T_j (K)"); ax2.set_title("Junction temperature vs. time")
ax2.grid(True, alpha=0.3)

Tvol = Tfield.reshape(mesh.nz, mesh.ny, mesh.nx)
X, Y = np.meshgrid(np.linspace(0, mesh.Lx*1e6, mesh.nx),
                   np.linspace(0, mesh.Ly*1e6, mesh.ny))
ax3 = fig.add_subplot(2, 2, 3)
im = ax3.imshow(Tvol[-1], origin="lower", cmap="inferno",
                extent=[0, mesh.Lx*1e6, 0, mesh.Ly*1e6])
ax3.set_xlabel("x (um)"); ax3.set_ylabel("y (um)")
ax3.set_title("2-D temperature map (top surface)")
fig.colorbar(im, ax=ax3, shrink=0.8)

ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.plot_surface(X, Y, Tvol[-1], cmap="inferno")
ax4.set_xlabel("x (um)"); ax4.set_ylabel("y (um)"); ax4.set_zlabel("T (K)")
ax4.set_title("3-D temperature surface")

fig.suptitle("Coupled HICUM + 3-D transient heat-diffusion simulation "
             "(V_BE=0.9 V, V_CE=2 V)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_coupled_sim.png"), dpi=120)
print("saved results_coupled_sim.png")
