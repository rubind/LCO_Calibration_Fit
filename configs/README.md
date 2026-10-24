# Filter configurations

Each YAML file is a standalone production configuration. Paths are relative to this directory, outputs go to `../runs`, wavelengths are vacuum Angstroms internally, and tqdm progress bars are enabled. Every configuration fits a zero-mean per-star offset hierarchy with `star_correlated_scatter: {scale: 0.03}` plus the independent `extra_scatter` term; both values are half-normal prior scales in magnitudes. The per-star offsets are marginalized analytically during sampling and reconstructed conditionally for output products.

All configurations use `model.atmosphere_mode: site_specific`. The first three
characters of each image's instrument prefix are resolved through
`../site_mapping.txt`, selecting one of the five unit-airmass transmission
curves in `passband_ingredients`. Those curves use nanometer wavelengths and
fractional transmission; each observation uses its site's `T1**airmass` curve.

Any configuration can set `model.passband_mode: fixed` to hold the nominal
passband exactly fixed while fitting only the instrument zeropoint hierarchy,
the per-star correlated scatter, and the point-to-point extra scatter.

`U_leak.yaml`, `up_leak.yaml`, `gp_leak.yaml` and `rp_leak.yaml` select the
native `shift_width_leak` mode.
They fit camera shifts, widths and leak scales with filter-level hyperparameters;
the main CLI accepts them directly. Out-of-band means >4500 Å for U/u′ and
>8000 Å for g′/r′, in the nominal filter frame. Their `leak` block uses
`main_band_blue_angstrom: 0.0` to retain every bluer wavelength. The grids are
1200–20000 Å for U, 2500–14000 Å for u′, and 2700–14000 Å for g′/r′ to contain
the full shifted and stretched tabulated tails. Saved exploratory runs retain
the original two-sided windows and two-chain, 300-warmup/300-draw settings in
their resolved configs. See the main README for model definitions, priors and
commands.

| Filter | Pivot (A) | Warp scale (A) | Matching rows | Used rows | Targets | Instruments | Grid delta (mmag) |
|---|---:|---:|---:|---:|---:|---:|---:|
| U  | 3600 | 800  | 126 | 126 | 10 | 11 | 0.192 |
| B  | 4400 | 1000 | 90  | 90  | 9  | 10 | 0.060 |
| V  | 5400 | 900  | 96  | 94  | 8  | 10 | 0.108 |
| R  | 6500 | 1400 | 89  | 89  | 10 | 10 | 0.251 |
| I  | 8400 | 2100 | 93  | 93  | 10 | 10 | 0.021 |
| up | 3550 | 700  | 118 | 118 | 11 | 11 | 0.256 |
| gp | 4750 | 1000 | 105 | 105 | 12 | 11 | 0.140 |
| rp | 6250 | 1400 | 98  | 97  | 13 | 11 | 0.010 |
| ip | 7500 | 1300 | 87  | 85  | 11 | 10 | 0.081 |
| zs | 8650 | 1000 | 78  | 78  | 12 | 10 | 0.020 |

All grid deltas are below the configured 0.5 mmag tolerance. V and ip each reject two rows with zero flux uncertainty, and rp rejects one. The U configurations use 1 Angstrom spacing because the new high-resolution site atmospheres gave a 0.518 mmag coarse/fine difference at 2 Angstrom spacing. The extended BD-11_3759 spectrum gives complete I-band coverage, and both `I.yaml` (`shift_only`) and `I_2.yaml` (`shift_width`) pass validation. Insufficient spectral coverage is a hard error and no target is silently excluded.

The response-support threshold is `1e-3`, chosen above the measured positive/negative digitization-noise floor in the tabulated filter curves. Grid containment is evaluated on the transformed filter multiplied by fixed aluminum², fixed CCD QE, and the pointwise least-extincted atmosphere among the selected observations and sites; the atmosphere therefore sets the blue edge and QE sets the red edge. Most wavelength grids remain 2700--11000 A. U_3 and up_3 use 2650--14200 A to pad both physical component edges. The I configurations use 2700--15000 A; `I_2.yaml` has numerical width guards 0.45--1.45 so its posterior no longer contacts the former 0.6 boundary. R and R_2 use a 2700--12000 A grid, a +/-1000 A numerical shift guard, `Normal(0, 300 A)` for the population mean shift, and `HalfNormal(100 A)` for its population scale. The ip_2 red grid endpoint is also extended to 12000 A. These broad guards are numerical safety limits, not substitutes for the scientific priors.

All shift-width and shift-width-warp configurations use 2000 warmup draws, `target_accept_prob: 0.95`, and a dense mass matrix. R uses the same sampler settings because its previous posterior contacted the old shift boundary and diverged. The 2-micron BD-11_3759 extension gives complete coverage for I, I_2, and I_3. All standalone filter configurations now pass complete input, response-envelope, spectrum-coverage, and grid-convergence validation.

Validate or run one filter from the project root:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit validate --config configs/rp.yaml --filter rp

PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit fit \
  --config configs/rp.yaml --filter rp --run-label rp-production-seed12345
```
