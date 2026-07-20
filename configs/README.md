# Filter configurations

Each YAML file is a standalone production configuration. Paths are relative to this directory, outputs go to `../runs`, wavelengths are vacuum Angstroms internally, and tqdm progress bars are enabled. Every configuration fits a zero-mean per-star offset hierarchy with `star_correlated_scatter: {scale: 0.03}` plus the independent `extra_scatter` term; both values are half-normal prior scales in magnitudes. The per-star offsets are marginalized analytically during sampling and reconstructed conditionally for output products.

| Filter | Pivot (A) | Warp scale (A) | Matching rows | Used rows | Targets | Instruments | Grid delta (mmag) |
|---|---:|---:|---:|---:|---:|---:|---:|
| U  | 3600 | 800  | 107 | 107 | 7  | 10 | 0.112 |
| B  | 4400 | 1000 | 73  | 73  | 6  | 9  | 0.060 |
| V  | 5400 | 900  | 80  | 80  | 7  | 9  | 0.104 |
| R  | 6500 | 1400 | 72  | 71  | 7  | 8  | 0.015 |
| I  | 8400 | 2100 | 84  | 84  | 8  | 10 | 0.022 |
| up | 3550 | 700  | 101 | 101 | 10 | 12 | 0.099 |
| rp | 6250 | 1400 | 78  | 78  | 10 | 11 | 0.029 |
| ip | 7500 | 1300 | 67  | 67  | 10 | 10 | 0.083 |
| zs | 8650 | 1000 | 58  | 57  | 8  | 9  | 0.015 |

All grid deltas are below the configured 0.5 mmag tolerance. R and zs each reject one row with zero flux uncertainty. The extended BD-11_3759 spectrum now gives complete I-band coverage, and both `I.yaml` (`shift_only`) and `I_2.yaml` (`shift_width`) pass validation. Insufficient spectral coverage is a hard error and no target is silently excluded.

The response-support threshold is `1e-3`, chosen above the measured positive/negative digitization-noise floor in the tabulated filter curves. Grid containment is evaluated on the transformed filter multiplied by fixed aluminum², fixed CCD QE, and the least-extincted observational atmosphere; the atmosphere therefore sets the blue edge and QE sets the red edge. Most wavelength grids remain 2700--11000 A. U_3 and up_3 use 2650--14200 A to pad both physical component edges. The I configurations use 2700--15000 A; `I_2.yaml` has numerical width guards 0.45--1.45 so its posterior no longer contacts the former 0.6 boundary. R and R_2 use a 2700--12000 A grid, a +/-1000 A numerical shift guard, `Normal(0, 300 A)` for the population mean shift, and `HalfNormal(100 A)` for its population scale. The ip_2 red grid endpoint is also extended to 12000 A. These broad guards are numerical safety limits, not substitutes for the scientific priors.

All shift-width and shift-width-warp configurations use 2000 warmup draws, `target_accept_prob: 0.95`, and a dense mass matrix. R uses the same sampler settings because its previous posterior contacted the old shift boundary and diverged. The 2-micron BD-11_3759 extension gives complete coverage for I, I_2, and I_3. All standalone filter configurations now pass complete input, response-envelope, spectrum-coverage, and grid-convergence validation.

Validate or run one filter from the project root:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit validate --config configs/rp.yaml --filter rp

PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit fit \
  --config configs/rp.yaml --filter rp --run-label rp-production-seed12345
```
