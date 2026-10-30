# SiGe HBT: electro-thermal coupling and Bayesian calibration

Reference code for my paper *Bayesian optimization study of SiGe HBT based on HICUM model and unsteady heat conduction* (manuscript in revision). It couples a reduced, temperature-aware HICUM model of a SiGe heterojunction bipolar transistor to a 3D transient heat-conduction solver, and calibrates the model parameters with Bayesian optimization.

## Files

- `thermal_solver.py`: 3D transient heat diffusion with finite volumes and implicit (backward Euler) time stepping, so it is unconditionally stable. Robin boundary conditions with a heat-sink bottom face and near-adiabatic sides and top, so vertical conduction (the thermal conductivity `k`) sets the thermal resistance.
- `hicum_model.py`: reduced HICUM electrical model. Transfer current uses the normalized base charge `q_b` (Early effect plus high injection), temperature laws for `Is`, `R` and `Vt` are dimensionally consistent, and the base current has an ideal and a non-ideal (n≈2) part.
- `coupled_solver.py`: monolithic Newton electro-thermal solver. The large thermal block is eliminated with a Schur complement on the 2×2 electrical block. Steady-state and transient drivers.
- `bayes_opt.py`: Gaussian-process surrogate (scikit-learn, Matérn 5/2) with a hand-written Expected Improvement acquisition and per-dimension log scaling.
- `twin_experiment.py`: validation on synthetic data (below).
- `calibrate_real.py`: calibration against silicon-calibrated reference curves of the IHP SG13G2 transistor (below).
- `make_fig53.py`: the four-panel coupled-simulation figure (Section 5.3 of the paper).
- `ihp_ref/`: model files from the IHP SG13G2 open PDK (Apache-2.0, original headers kept, license text in `ihp_ref/LICENSE-Apache-2.0.txt`) and the ngspice decks and output used as reference data.

## Run

```
pip install -r requirements.txt
python thermal_solver.py      # heat-diffusion check
python hicum_model.py         # Gummel curve and temperature sensitivity
python coupled_solver.py      # coupled operating points and transient self-heating
python bayes_opt.py           # BO on a 3D test function (global minimum about -2)
python twin_experiment.py     # synthetic calibration -> results_calibration.png, results_thermal_field.png
python calibrate_real.py      # IHP calibration -> results_real_calibration.png
python make_fig53.py          # -> results_coupled_sim.png
```

The reference curves in `ihp_ref/` are already generated. To regenerate them, install [ngspice](https://ngspice.sourceforge.io/download.html) (version 46 was used) and run `ngspice -b generate_ref.sp` from inside `ihp_ref/`.

## Twin experiment

1. Fix known parameters `θ_true = [IS0, XTI, RE0, k]`.
2. Run the coupled model to produce a DC Gummel curve `Ic(Vbe)`, low-bias Gummel curves at several ambient temperatures and a transient self-heating curve `Tj(t)`, then add measurement noise (3% on currents, 0.3 K on temperature).
3. Throw away `θ_true` and let Bayesian optimization minimize `L(θ) = Σ wᵢ‖y_sim(θ) − y_meas‖²` against the noisy data.
4. Check that the parameters come back: `IS0`, `RE0` and `k` were recovered to within about 3–7% and `XTI` to about 15%, limited by a remaining correlation between `IS0` and `XTI`.

All data in this experiment is synthetic and labelled that way in the figures.

## Which parameters can be identified

Not every parameter can be recovered from every dataset:

- `IS0` is fixed by the low-bias current level at 300 K.
- `XTI` (the temperature exponent of `Is`) only becomes identifiable when low-bias currents are measured at several ambient temperatures; at a single temperature `IS0` and `XTI` trade off against each other.
- `k` is only identifiable when conduction dominates the thermal resistance (heat-sink bottom, adiabatic sides). Under uniform convective cooling the device is nearly isothermal and `k` has no visible effect.
- `RB0` was dropped: the base current is so small that `Ib·RB` is negligible.

This is why industrial HICUM extraction uses multi-temperature I–V together with thermal measurements.

## Calibration against IHP SG13G2 reference data

The IHP SG13G2 open PDK ships a VBIC model card for its `npn13G2` SiGe HBT, extracted from measured wafers (lot EDJ802). VBIC level 9 runs natively in ngspice. `ihp_ref/generate_ref.sp` produces forward Gummel curves at −40, 27, 85 and 125 °C and output characteristics at four base voltages. `calibrate_real.py` fits the reduced HICUM model, plus a lumped self-heating loop `Tj = Tamb + Rth·P`, to those curves with Bayesian optimization.

With the non-ideal base current and physically scaled base-current temperature laws (ideal ∝ ni², non-ideal ∝ ni), a single 8-parameter fit reproduces the multi-temperature Gummel `Ic` (including −40 °C), the base current across temperature and the output characteristics. The log-domain objective went from 6.5 to 0.029. The fitted values (IS0 ≈ 6.1e-18 A, XTI ≈ 4.2, Eg ≈ 0.98 eV, IBES ≈ 3.1e-20 A, IBEN ≈ 3.7e-17 A, IKF ≈ 0.94 mA, RE0 ≈ 19.7 Ω, Rth ≈ 4.0e3 K/W) are physically sensible for this device. The remaining mismatch is the Early slope of the highest output curve, because `VAF` was held fixed. This covers DC behavior across temperature; RF behavior (fT, S-parameters) would need capacitances and transit time.

## Outputs

- `results_calibration.png`: Gummel fit, transient `Tj(t)` fit, BO convergence and parameter recovery (twin experiment).
- `results_thermal_field.png`: top-surface and vertical-slice temperature fields at the hottest operating point.
- `results_real_calibration.png`: fit against the IHP reference curves.
- `results_coupled_sim.png`: coupled electro-thermal simulation (Section 5.3).

## Citation

If you use this code, please cite it (see `CITATION.cff`). The paper is still in revision.

## License

MIT for my code. The files from the IHP SG13G2 PDK in `ihp_ref/` are under the Apache License 2.0.
