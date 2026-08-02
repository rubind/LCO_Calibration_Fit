# passband-fit

`passband-fit` fits one LCO photometric filter at a time against CALSPEC spectrophotometric standards. It combines a nominal filter curve, the supplied aluminum-squared reflectivity, CCD quantum efficiency, and a fixed site-specific atmospheric transmission curve, then partially pools passband-shape corrections across the telescope/camera systems present in the selected observations.

The detectors are photon counting. All wavelengths are converted to vacuum Angstroms and synthetic integrals use the required wavelength factor. Both Vega and AB reference spectra are supported. Each supplied atmosphere table contains wavelength in nanometers and unit-airmass transmission as a fraction. The image prefix selects the site curve through `site_mapping.txt`, and an observation at airmass `X` uses `T1**X`.

## Model

For instrument `j`, the nominal filter is remapped as

```text
Fj(lambda) = remap(filter, shift[j], width[j])
             * exp(-warp[j] * (lambda - pivot) / warp_scale).
```

The source coordinate used to remap the nominal response is

```text
lambda_source = pivot + (lambda - pivot - delta_lambda[j]) / width[j].
```

Thus a positive shift moves the observed passband redward, and the shift remains an exact observed-frame displacement even when width is fitted. The optional warp is

```text
exp(-warp[j] * (lambda - pivot) / warp_scale).
```

The atmosphere-free instrumental response is

```text
Pj(lambda) = Fj(lambda) * aluminum_squared(lambda) * QE(lambda).
```

Thus the fitted transform changes the filter while aluminum and CCD QE remain
fixed; in particular, the QE continues to define the red cutoff. For each
observation, this response is multiplied by the atmosphere at that
observation's airmass, which defines the blue cutoff. Instrument zeropoints and
enabled shift, log-width, and warp parameters are modeled hierarchically. Two
positive scatter terms are fitted per filter: independent point-to-point extra
scatter and a per-star correlated scatter.

For star `s`, the shared residual model is

```text
star_offset[s] ~ Normal(0, sigma_star).
```

Its population mean is fixed at zero so it cannot duplicate the mean instrument
zeropoint. The sampler analytically marginalizes these offsets, giving the exact
covariance

```text
C[i,k] = (sigma_m[i]^2 + sigma_extra^2) * I(i=k)
         + sigma_star^2 * I(star[i]=star[k]).
```

Thus repeated observations of one star share a grey, band-specific residual,
while measurements of different stars do not. This diagonal-plus-rank-one
calculation avoids sampling the funnel-prone star coordinates without using a
dense covariance matrix. Physical star offsets are drawn afterward from their
exact conditional Gaussian posterior for tables and plots. `sigma_star` has
the configurable prior `priors.star_correlated_scatter` (default half-normal
scale 0.03 mag).

### Passband model modes

Set the mode in a YAML configuration under `model`:

```yaml
model:
  passband_mode: shift_only
```

The supported values are:

- `fixed`: applies the nominal filter, aluminum, CCD QE, and atmosphere with no
  fitted shift, width, or warp parameters. Only the instrument zeropoint
  hierarchy, per-star correlated scatter, and point-to-point extra scatter are
  fitted.

- `shift_only`: fits an instrument-specific wavelength shift
  `delta_lambda[j]` in Angstroms. Width remains 1 and warp remains 0.

- `shift_width`: fits the wavelength shift and a positive width scale
  `width[j] = exp(eta[j])`. A width above 1 broadens the response around the
  configured pivot; a width below 1 narrows it.

- `shift_width_warp`: fits shift and width plus `warp[j]`, which multiplies the
  response by `exp(-warp[j] * (lambda - pivot) / warp_scale)`. A positive warp
  suppresses the red side relative to the blue side.

Every mode fits instrument zeropoints, marginalized per-star offsets and their
common population scatter, and the common point-to-point extra-scatter term.
The enabled passband parameters and zeropoints are partially pooled across
instruments with hierarchical population distributions. In `fixed` mode the
shape prior entries and numerical shift/width/warp bounds remain valid config
fields but are unused. More flexible modes require correspondingly stronger
wavelength/color leverage from the standards.

## Inputs

The included `example_config.yaml` documents the schemas and points to the files in this workspace. Observation rows contain target, filter, image filename, airmass, flux, flux uncertainty, and optional PSF residual RMS. The telescope/camera ID is parsed from image prefixes such as `lsc1m004-fa03`; the leading three characters are the site code. `site_mapping.txt` maps `lsc`, `cpt`, `elp`, `coj`, and `tfn` to CTIO, Sutherland, McDonald, Siding Spring, and Tenerife-ish, respectively. Missing observed prefixes or mapped sites without configured curves are hard validation errors.

Every selected stellar reference spectrum, and the Vega spectrum for Vega
magnitudes, must cover the complete non-negligible response envelope allowed by
the configured shift, width, and warp bounds. Validation stops with an error if
any blue or red tail is missing; the message reports the available range, the
required vacuum-Angstrom range, and the missing extent. Spectra are never
silently extrapolated or dropped from the fit. Extend a deficient tail with an
appropriate stellar-library spectrum before rerunning validation.

Grid and stellar-spectrum coverage are evaluated on the union of the complete
responses across the observed sites: transformed filter times fixed aluminum,
fixed QE, and the least-extincted relevant site atmosphere. Every configured
site atmosphere must be negligible at its blue table edge and QE negligible at
its red table edge before zero fill is allowed.
Consequently, filter warping alone cannot force coverage into wavelengths where
the atmosphere or CCD has essentially no response.

Standalone production configurations and their validation results for every other filter are documented in `configs/README.md`.

Set `sampler.progress: true` to display NumPyro's warmup/sampling progress bars. Smoke runs keep progress disabled because they finish quickly.

Sampling uses NumPyro NUTS. With `sampler.chain_method: parallel`, the CLI configures one virtual JAX CPU device per chain before JAX starts; this avoids Python multiprocessing and allows the four chains to run concurrently on Apple Silicon. If JAX was already initialized in an interactive process with too few devices, the sampler falls back to sequential chains and records that fact in the diagnostics.

The differentiable likelihood retains the exact configured-grid calculation: every NUTS state remaps the filter, multiplies the fixed aluminum, QE, and observation-specific atmosphere, and evaluates both trapezoidal photon-counting integrals. There is no passband emulator, color-term approximation, or effective-wavelength approximation in the likelihood.

Flux is assumed to be an exposure-normalized count rate on a consistent gain/aperture convention. The PSF residual RMS is retained only as an auxiliary diagnostic. Nonpositive flux, nonpositive uncertainty, and low-SNR observations are counted explicitly; malformed or nonfinite selected-filter rows fail validation.

## Commands

Run directly from the checkout:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit validate --config example_config.yaml --filter gp
PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit fit --config example_config.yaml --filter gp --run-label gp-smoke --smoke
PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit summarize --run runs/gp/gp-smoke
PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit plot --run runs/gp/gp-smoke
```

An editable install also provides the `passband-fit` command. `validate` performs all loading, conversion, selection, coverage, and grid checks without sampling. NumPyro runs start in a new nonempty-safe directory; legacy emcee/HDF5 runs are not resumable as NumPyro chains. Plot and summarize commands consume persisted artifacts without rerunning inference. Smoke output verifies execution and is never a convergence or scientific-result claim.

To regenerate plots for every completed run without rerunning sampling, use the
top-level batch script:

```bash
/opt/anaconda3/bin/python rerun_all_plots.py
```

The script discovers `runs/*/*/manifest.json`, skips incomplete run directories
that have no `chain.npz` or legacy `chain.h5`, reports progress and failures, and
returns a nonzero exit status if any plottable run fails. Use `--dry-run` to list
what would be processed or `--stop-on-error` to stop at the first failure.

To compare the fitted mean zeropoint hyperparameter across all completed runs,
including its 16th--84th percentile interval, run:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python plot_mean_zeropoints.py
```

This writes `runs/mean_zeropoints_all_runs.png` and a matching CSV summary. The
plot uses Vega=0 zeropoints for U/B/V/R/I and AB zeropoints for up/gp/rp/ip/zp;
the internal `zs` run directory is displayed as `zp` in this cross-run figure.

To compare the source-independent mean effective wavelength of the
population-level passband across completed runs, use:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python plot_mean_effective_wavelengths.py
```

This writes `runs/mean_effective_wavelengths_all_runs.png` and a matching CSV.
For each posterior draw, the filter is transformed with the population-mean
shift, log-width, and warp hyperparameters, then multiplied by fixed aluminum²
and CCD QE. The quoted vacuum-Angstrom value is the atmosphere-free throughput
centroid `integral(lambda*P)/integral(P)`. Thus each run contributes one
population-passband wavelength rather than one wavelength per instrument.

To compare the individual site/camera offsets from their filter populations in
the shift-only runs, use:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python plot_shift_only_camera_offsets.py
```

This writes `runs/shift_only_camera_offsets.png` and a matching CSV. Each filter
is one row, with zeropoint and wavelength-shift offsets in separate columns.
Labels identify offsets whose posterior sign probability exceeds the
Gaussian-equivalent threshold set by `--outlier-sigma` (default 2.5).

To generate a population-level passband posterior plot inside every completed
run directory, use:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python plot_population_passbands.py
```

Each plot and matching NPZ use posterior draws of the enabled population-mean
shape parameters (`mu_delta_lambda`, `mu_eta`, and `mu_a`) only. Population
dispersion parameters and individual-instrument deviations are excluded. The
transformed filter is multiplied by fixed aluminum² and CCD QE, with atmosphere
excluded, and every draw is normalized to unit peak so grey throughput
normalization and zeropoint information do not enter the plotted 68% interval.
The regular `passband-fit plot` command and `rerun_all_plots.py` also regenerate
this population-passband product for each run.

To render the publication-style Daft probabilistic graphical model of the
hierarchical calibration model, run:

```bash
/opt/anaconda3/bin/python make_calibration_pgm.py
```

This writes PNG and vector PDF versions under `figures/`. The diagram includes
the camera and standard-star plates, passband-shape and zeropoint hierarchies,
the analytically marginalized star offset, and the complete photon-counting
wavelength integral.

To make a LaTeX site-by-standard observing grid from a photometry table, run:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python make_observation_grid.py \
  all_phot_filtered.txt \
  --output observed_photometry_site_grid.tex
```

Rows are sorted by synthetic B-V. Each site column lists the filters in which
that standard was observed. The sorting color uses the nominal combined B and V
responses (filter, aluminum, CCD QE, and the mean configured unit-airmass site atmosphere) in the
Vega-zero convention. Filter names and numerical values are printed in math
mode, with `up`/`gp`/`rp`/`ip` rendered as $u^\prime$/$g^\prime$/$r^\prime$/$i^\prime$
and the internal `zs` label rendered as $z_s$. Filter labels outside the requested
list are ignored with a warning. Use `--help` to override the B/V configurations,
table format, included filters, caption, or table label.

To examine repeat observations of one Landolt star by site/camera and over
time, run:

```bash
/opt/anaconda3/bin/python analyze_landolt_star.py \
  all_phot_filtered_LandoltStar.txt
```

The table must include the usual photometry columns plus `MJD`, and its fluxes
are assumed to have already been intercalibrated using field stars. The script
therefore fits no nightly zeropoints and no airmass terms. Repeated exposures
are averaged per camera and observing date only to avoid treating them as
independent evidence. It fits both a common long-term slope per filter (with
separate camera intercepts) and individual camera slopes. By default, an
individual slope requires at least six dates spanning at least 180 days.
Trend flags require both 3-sigma nominal significance and a
Benjamini-Hochberg false-discovery-rate q-value below 0.05. The
`landolt_long_term_trends/` output contains the filter and camera trend tables,
the diagnostic plot, and a plain-text report.

### Extending a CALSPEC tail with stellar templates

`scripts/extend_calspec.py` creates a new FITS spectrum without modifying the
input CALSPEC file or any measured CALSPEC row. It recursively searches an
ASCII-template directory, compares every template with the CALSPEC overlap,
and selects the template with the lowest median absolute deviation (MAD) in
overlap magnitude residuals independently for each requested
blue/red tail. During matching it fits a robust low-order polynomial to
`log(CALSPEC/template)`. This is a very smooth multiplicative correction that
can absorb normalization, continuum tilt, and gentle extinction-like curvature
without fitting away individual spectral features.

For the currently missing I-band red tail, a typical command is:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python scripts/extend_calspec.py \
  CALSPEC_spectra/bd11d3759_stis_003.fits \
  /path/to/ascii_stellar_templates \
  CALSPEC_spectra/bd11d3759_stis_003_extended_2um.fits \
  --max-wavelength 20000 \
  --continuum-degree 0 \
  --template-wavelength-unit angstrom \
  --template-wavelength-convention vacuum
```

ASCII templates default to wavelength column 0 and flux column 1; use
`--template-wave-column`, `--template-flux-column`, `--delimiter`, and
`--template-wavelength-unit` for other layouts. Air wavelengths can be
converted with `--template-wavelength-convention air`. At least one of
`--min-wavelength` or `--max-wavelength` is required.

For a long extrapolation, prefer `--continuum-degree 0` unless the diagnostic
plot establishes that extrapolating a fitted tilt or curvature remains
physical. A local quadratic continuum correction can behave pathologically far
outside its overlap interval even when its overlap MAD is small.

The output SCI table adds `MODEL_EXT`, which is false for original CALSPEC rows
and true for synthetic rows. Synthetic rows have `DATAQUAL=2`, `TOTEXP=0`, and
NaN statistical/systematic errors so they cannot be mistaken for measurements.
The script also writes an extension plot, a JSON provenance record, and a CSV
ranking of all candidate templates. Inspect these diagnostics before changing
`star_spectra_mapping.txt` to use the extended file, then rerun `validate`.

## Outputs

Each run stores the resolved configuration and input hashes, selection and index mappings, nominal component and combined curves, the authoritative NumPyro chain in `chain.npz`, flattened samples, parameter summaries and correlations, nominal and fitted synthetic magnitudes and residuals, posterior passband products, posterior-predictive summaries, plots, and a machine-readable manifest. The physical summaries include `sigma_star` and conditionally reconstructed `star_offset[target]` values, and `plots/forest_star_offset.png` shows their intervals. `nominal_curves.csv` contains every relevant site atmosphere plus a compatibility column holding their equally instrument-weighted mean. `zeropoint_conventions.json` records each instrument's nominal unit-airmass AB-minus-Vega conversion and uses their equal-weight mean for the population-mean zeropoint label; every likelihood integral uses the exact observation's site and airmass. The internal magnitude bookkeeping constant is not included. `diagnostics.json` and `diagnostics_by_parameter.csv` save R-hat, effective sample size, acceptance probabilities, divergences, tree-depth information, and the effective chain execution method. `boundary_diagnostics.json` reports posterior proximity to numerical shift/width/warp limits and the wavelength-grid endpoints.

## Verification

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -m "not slow"
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -m slow
```

The fast suite emphasizes deterministic unit/sign checks. Slow tests use fixed-seed mock recovery and declared tolerances; no short smoke chain is described as converged.

## First-version assumptions and limitations

Observations are conditionally independent given the analytically marginalized per-star offsets. One filter is fitted per run, so the star offsets and their population scatter are separate for each band. A fixed mean atmosphere is selected for each site and scaled as `T1**airmass`; the atmospheric curves themselves are not fitted. There are no night terms, atmospheric nuisance parameters, robust likelihood, time-dependent correlated-repeat model, or propagated wavelength-dependent CALSPEC spectral covariance. One common point-to-point extra-scatter parameter and one common per-star correlated-scatter parameter are used per filter, and direct numerical integration is performed during likelihood evaluation.

Differences between a site's fixed mean atmosphere and the actual observing-night atmosphere can transfer extinction errors into inferred passband shape. Residual trends versus airmass, site, target, color, and instrument must therefore be inspected. The current star offset is grey within a band; it does not model wavelength-dependent CALSPEC errors. Future versions can add time-dependent atmospheres, night zeropoints, time-dependent correlated repeats, instrument scatter, spectral-shape nuisance terms, robust likelihoods, atmospheric nuisance parameters, and faster photometry emulators.
