# Physical model and three-replicate mean data

Open `quanouhe1_KunsatSe3.ipynb` and run its cells in order to start the baseline
three-parameter inversion. The notebook contains the physical equations, inverse
model, full training configuration, and execution cell. It starts from Keos=0.5,
RH=35, and pH50=7 without restoring an inversion checkpoint. Results are written
to the ignored `results` directory.

For each case-location-time combination, the `data` directory reports the arithmetic mean
of three independent experimental replicates and their sample standard deviation.
Observed means are reported to two decimal places; measurement SD values retain
their recorded precision.

| File | Role | Rows |
|---|---|---:|
| `pH_calibration.csv` | pH inversion target | 968 |
| `TotalPb_calibration.csv` | total-Pb inversion target | 8 |
| `pH_validation.csv` | pH external validation | 726 |
| `TotalPb_validation.csv` | total-Pb external validation | 6 |

The model fits the `*_obs` column. `pH_measurement_sd` and
`TotalPb_measurement_sd` are the observation standard deviations used for
weighted residuals and uncertainty analysis.

The retained baseline estimate is Keos=0.8177583814, RH=19.45045090, and
pH50=3.322480917.
