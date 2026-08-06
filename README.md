# PINN three-parameter inversion

## Contents

- `model_and_data`: the physical-model notebook and the four CSV files
  actually used as calibration or validation inputs.
- `uncertainty_analysis`: optimizer stability, uncertainty, identifiability,
  and global-sensitivity analyses.

All measurement tables contain only coordinates/metadata, the observed value,
and an explicitly named measurement standard deviation.

Each observed value is the arithmetic mean of three independent experimental
replicates. The corresponding measurement-SD column is the sample standard
deviation across those three replicates.

Observed pH and total-Pb means are reported to two decimal places. Measurement
SD values retain their recorded precision.

The measurement uncertainty columns are:

- `pH_measurement_sd` for pH;
- `TotalPb_measurement_sd` for total Pb, in the same units as
  `TotalPb_obs_mol_m3_bulk`.

Calibration observations enter the inversion. Validation observations are used
only for final reporting.
