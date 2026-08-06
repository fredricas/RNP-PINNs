# Formal uncertainty and sensitivity analysis

The analysis uses the physical-model notebook, three-replicate mean data, fitted
baseline, and staged-training checkpoints included in this directory.

## Methods

1. Ten-start optimizer-stability analysis: the original start plus nine
   Latin-hypercube starts spanning the stated parameter bounds.
2. Fifty heteroscedastic Gaussian parametric-bootstrap refits.
3. Profile likelihood with nuisance parameters re-optimized.
4. Morris global sensitivity with six levels and ten trajectories.
5. Direct 9-by-9 objective-function grids without a surrogate surface.

Only calibration observations are fitted. Validation observations remain
out-of-fit. The code reads `pH_measurement_sd` and `TotalPb_measurement_sd`.

Each `*_obs` value is the arithmetic mean of three independent experimental
replicates. Each measurement-SD value is the sample standard deviation across
those three replicates. Observed means are reported to two decimal places, while
measurement SD values retain their recorded precision. For optimizer-stability
analysis, neural-network weights are initialized from a common checkpoint,
while Keos, RH, and pH50 retain their designated initial values.

## Run

Create a 64-bit Python 3.12 environment and install the pinned dependencies:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Run the analysis driver directly from a terminal:

```powershell
.\venv\Scripts\python.exe .\run_uncertainty_analysis.py --mode formal --sections stability,bootstrap,profile,morris,surface --workers 2
```

Results are written beneath `outputs`.
