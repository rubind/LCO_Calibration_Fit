Build a Python package and command-line program for fitting the effective photometric passbands of Las Cumbres Observatory 1-meter telescope/camera systems using spectrophotometric CALSPEC standards.

## Scientific objective

The current observation table contains 12 unique LCO telescope/camera combinations. The implementation must discover this count from the data rather than hard-code it. The standards span a broad range of spectral types, from hot stars to a brown dwarf. The goal is to improve the estimated effective passband of one named filter at a time by comparing observed magnitudes with synthetic magnitudes computed from the CALSPEC spectra.

Each nominal effective response consists of:

[
P_{j,f}(\lambda, X)
===================

F_f(\lambda),
A(\lambda)^2,
Q(\lambda),
T_{\rm atm}(\lambda,X),
]

where:

* (j) identifies one telescope/camera system;
* (f) is the selected filter;
* (F_f(\lambda)) is the nominal filter transmission;
* (A(\lambda)) is aluminum mirror reflectivity, applied twice;
* (Q(\lambda)) is the nominal CCD quantum efficiency;
* (T_{\rm atm}(\lambda,X)) is atmospheric transmission at airmass (X).

The filters, mirrors, and CCDs are nominally the same across the systems, but small system-to-system passband differences are expected. These differences should be modeled hierarchically.

For the first implementation, use one fixed tabulated KPNO atmospheric transmission curve. Evaluate the atmosphere separately at the airmass of each observation. Design the code so that site-specific atmosphere files can be added later without changing the model architecture.

Fit only one filter per run. The command-line interface should require the user to specify the filter.

## Input observations

The observation table has one line per observation and contains at least:

* telescope/camera system identifier;
* target identifier;
* site identifier;
* filter identifier;
* airmass;
* measured flux;
* measured flux uncertainty.

There may be repeated observations of the same target with the same telescope/camera. Treat observations as conditionally independent given a fitted shared per-star offset; marginally, repeated measurements of a star are correlated.

The measured fluxes use an instrumental zeropoint convention of 24:

[
m_{\rm obs}
===========

24 - 2.5\log_{10}(F_{\rm obs}).
]

Convert flux uncertainties to magnitude uncertainties using:

[
\sigma_{m,i}
============

\frac{2.5}{\ln 10}
\frac{\sigma_{F,i}}{F_i}.
]

Apply a configurable signal-to-noise cut:

[
F_i/\sigma_{F,i} > {\rm SNR}_{\rm min},
]

with default:

[
{\rm SNR}_{\rm min}=10.
]

Reject nonpositive fluxes before converting to magnitude.

The program should report:

* total number of rows read;
* number matching the selected filter;
* number rejected because of nonpositive flux;
* number rejected by the signal-to-noise cut;
* final number used;
* number of observations per target;
* number of observations per telescope/camera system.

## Required input files

Support configurable paths for:

1. Observation table.
2. Nominal filter transmission curves.
3. CCD response curve.
4. Aluminum reflectivity curve.
5. Atmospheric transmission curve.
6. Directory of CALSPEC spectra.
7. Mapping from target name to CALSPEC spectrum filename.
8. Mapping from telescope/camera identifier to site.
9. CALSPEC Vega spectrum.
10. Configuration file containing model and prior choices.

Use simple documented formats such as CSV, whitespace-delimited ASCII, YAML, or JSON.

Do not hard-code target names, instrument names, filters, wavelength ranges, or file locations.

## Wavelength units and interpolation

Use Angstroms internally for all wavelengths.

Convert input wavelength units explicitly when loading each file. The configuration should specify the wavelength unit of every input curve.

Interpolate all curves onto a common configurable wavelength grid for the selected filter. The grid should adequately sample the filter edges and spectral features. A fixed grid spacing of approximately 1–5 Angstroms is acceptable initially, but make it configurable.

Use linear interpolation by default.

Outside the tabulated wavelength range of a throughput component, use zero throughput unless configured otherwise.

Do not extrapolate CALSPEC spectra. Require adequate coverage over the full
wavelength region with non-negligible effective response. If any selected
target or Vega reference lacks coverage, stop validation and report its
available range, required range, and missing blue/red tail; never silently
exclude the target.

The nominal passband files may contain tiny, uncommon negative values caused by digitization. These can be left unchanged or clipped to zero when loading. Do not build any special log-throughput machinery.

## Atmospheric transmission

Assume the supplied atmospheric curve represents transmission at unit airmass:

[
T_{\rm atm}(\lambda,X)
======================

T_{\rm atm}(\lambda,1)^X.
]

For numerical stability, values should be bounded to a physically meaningful range before exponentiation, for example:

[
0 \le T_{\rm atm}(\lambda,1) \le 1.
]

Document this assumption clearly.

The architecture should permit a later extension in which each site maps to a different atmosphere file.

## Photon-counting synthetic photometry

The detectors count photons and the measured signals are in electrons. Therefore use photon-counting weighting.

For a source with spectral flux density (S(\lambda)), define the photon-counting integral:

[
I[S,P]
======

\int \lambda,P(\lambda),S(\lambda),d\lambda.
]

The irrelevant constant factor (1/(hc)) cancels in all magnitude ratios and need not be included.

The overall multiplicative normalization of the passband is arbitrary and must cancel in the synthetic-magnitude definition. Do not fit or normalize an overall passband amplitude.

### Vega convention

When using Vega magnitudes, define:

[
m_{\rm syn,Vega}
================

-2.5\log_{10}
\left[
\frac{
\int \lambda,P(\lambda),S(\lambda),d\lambda
}{
\int \lambda,P(\lambda),S_{\rm Vega}(\lambda),d\lambda
}
\right].
]

This convention makes the supplied CALSPEC Vega spectrum exactly zero magnitude in every filter.

### AB convention

Also support AB magnitudes.

Implement the photon-counting AB magnitude rigorously. One acceptable expression is:

[
m_{\rm syn,AB}
==============

-2.5\log_{10}
\left[
\frac{
\int \lambda,P(\lambda),S_\lambda(\lambda),d\lambda
}{
\int \lambda,P(\lambda),S_{\lambda,{\rm AB}}(\lambda),d\lambda
}
\right],
]

where (S_{\lambda,{\rm AB}}) is the wavelength-space spectrum corresponding to a constant

[
f_\nu = 3631\ {\rm Jy}.
]

Construct (S_{\lambda,{\rm AB}}) using consistent units:

[
f_\lambda = f_\nu \frac{c}{\lambda^2}.
]

Be careful about the conversion between flux per unit centimeter and flux per Angstrom. Include unit tests validating the AB implementation against an independent calculation or a trusted synthetic-photometry package.

The magnitude system should be selected in the configuration as either `vega` or `ab`.

## Passband parameterization

Fit instrument-specific wavelength-shape modifications to the nominal instrumental passband.

The supported model modes should be:

1. `shift_only`
2. `shift_width`
3. `shift_width_warp`

The selected mode is specified in the configuration for the current filter.

Let:

* (\lambda_p) be a fixed pivot wavelength for the filter;
* (\Delta\lambda_j) be the coherent wavelength shift for instrument (j), in Angstroms;
* (w_j) be the width scale, with nominal value (w_j=1);
* (a_j) be the optional wavelength-warp parameter.

Define the wavelength remapping as:

\[
\lambda_{\rm source}
=
\lambda_p+
\frac{\lambda-\lambda_p-\Delta\lambda_j}{w_j}.
\]

Then evaluate the nominal filter at the remapped wavelength:

[
F_{j}^{\rm remap}(\lambda)
============================

F_f(\lambda_{\rm source}).
]

This convention makes (\Delta\lambda_j) an exact observed-frame displacement in Angstroms, independent of the width scale. For a nominal feature at (\lambda_0), the remapped feature appears at

\[
\lambda_{\rm feature}
=
\lambda_p+w_j(\lambda_0-\lambda_p)+\Delta\lambda_j.
\]

Check the sign convention carefully and document it. The desired interpretation is:

* positive (\Delta\lambda_j) shifts the observed passband toward longer wavelengths;
* (w_j>1) produces a broader passband;
* (w_j<1) produces a narrower passband.

Add an automated test using a simple Gaussian passband to verify these interpretations, including a case in which both (w_j\ne1) and (\Delta\lambda_j\ne0). The previous alternative, (\lambda_p+(\lambda-\lambda_p)/w_j-\Delta\lambda_j), is not used because it produces an observed-frame shift of (w_j\Delta\lambda_j) when width is also fitted.

For `shift_only`, fix:

[
w_j=1,\qquad a_j=0.
]

For `shift_width`, fit:

[
\Delta\lambda_j,\qquad w_j,
]

and fix:

[
a_j=0.
]

For `shift_width_warp`, fit all three:

[
\Delta\lambda_j,\qquad w_j,\qquad a_j.
]

### Wavelength warp

Use a pivoted exponential warp:

[
W_j(\lambda)
============

\exp
\left[
-a_j
\frac{\lambda-\lambda_p}{\lambda_{\rm scale}}
\right],
]

where (\lambda_{\rm scale}) is configurable and should normally be similar to the nominal filter width.

The modified non-atmospheric passband is:

[
P_j^{\rm inst}(\lambda)
=======================

F_{j}^{\rm remap}(\lambda)\,
W_j(\lambda)\,
A(\lambda)^2\,
Q(\lambda).
]

The remap and warp modify the filter curve only. Aluminum reflectivity and CCD
QE remain fixed multiplicative components, so in particular the QE retains its
physical red cutoff. The warp is intended to absorb smooth differences in the
effective filter shape without moving or exponentially amplifying the fixed QE
edge.

Do not add an amplitude parameter. Multiplicative passband normalization cancels in the reference-spectrum magnitude ratio.

For each observation (i) from instrument (j) at airmass (X_i), use:

[
P_i(\lambda)
============

P_j^{\rm inst}(\lambda),
T_{\rm atm}(\lambda,X_i).
]

The nominal non-atmospheric passband should initially be formed from:

[
P_0(\lambda)
============

F_f(\lambda),
A(\lambda)^2,
Q(\lambda).
]

## Photon-weighted mean wavelength

For each source and passband, compute the photon-weighted mean wavelength:

[
\langle\lambda\rangle_{\rm ph}
==============================

\frac{
\int \lambda^2 P(\lambda)S(\lambda),d\lambda
}{
\int \lambda P(\lambda)S(\lambda),d\lambda
}.
]

Compute this quantity using:

1. the original nominal passband;
2. the posterior fitted passband.

This will be used as the horizontal axis of residual plots.

## Observation model

For observation (i), let:

* (j[i]) identify its telescope/camera;
* (S_i) identify its CALSPEC spectrum;
* (X_i) be its airmass;
* (m_{{\rm syn},i}) be its synthetic magnitude under the selected magnitude system;
* (Z_j) be the additive zeropoint offset for telescope/camera (j).
* (d_s) be the band-specific shared residual for CALSPEC star (s).

Use:

[
m_{{\rm obs},i}
\sim
\mathcal N
\left(
m_{{\rm syn},i}+Z_{j[i]}+d_{S_i},
\sqrt{\sigma_{m,i}^2+\sigma_{\rm extra}^2}
\right).
]

Use one common extra-scatter parameter for the entire selected filter:

[
\sigma_{\rm extra}>0.
]

Use the non-centered per-star hierarchy

[
d_s=\sigma_{\rm star}z_s,\qquad z_s\sim\mathcal N(0,1),
]

with the population mean fixed at zero. This is equivalent, after marginalizing
the nuisance offsets, to adding (\sigma_{\rm star}^2) covariance between every
pair of observations of the same star. Retain the nuisance form in NumPyro to
avoid dense covariance matrices. Do not include night-level effects or a robust
likelihood in this version.

The zeropoint (Z_j) is an additive correction linking the observed 24-mag instrumental convention to the selected synthetic magnitude system.

## Hierarchical model

Each fitted passband parameter should be partially pooled across the telescope/camera systems present after selection.

### Coherent wavelength shifts

[
\Delta\lambda_j
\sim
\mathcal N
\left(
\mu_{\Delta\lambda},
\tau_{\Delta\lambda}
\right).
]

### Width parameters

It may be numerically preferable to model width in logarithmic form:

[
\eta_j = \ln w_j,
]

with:

[
\eta_j
\sim
\mathcal N
\left(
\mu_\eta,
\tau_\eta
\right),
]

and:

[
w_j=e^{\eta_j}.
]

This guarantees positive width.

### Warp parameters

[
a_j
\sim
\mathcal N
\left(
\mu_a,
\tau_a
\right).
]

### Zeropoints

[
Z_j
\sim
\mathcal N
\left(
\mu_Z,
\tau_Z
\right).
]

Fit all relevant population means and dispersions.

Only include the width hierarchy when width is enabled. Only include the warp hierarchy when the warp is enabled.

Use a non-centered parameterization where helpful:

[
\theta_j=\mu_\theta+\tau_\theta z_j,
\qquad
z_j\sim\mathcal N(0,1).
]

NUTS requires a differentiable posterior; use the non-centered variables to
improve hierarchical geometry while keeping the full photometric integrals in
the differentiated computation.

## Initial priors

All numerical priors must be configurable.

Choose initial defaults that are broad enough for exploration but physically reasonable.

Possible starting defaults are:

[
\mu_{\Delta\lambda}
\sim
\mathcal N(0,50^2\ {\rm \AA}^2),
]

[
\tau_{\Delta\lambda}
\sim
{\rm HalfNormal}(30\ {\rm \AA}),
]

[
\mu_\eta
\sim
\mathcal N(0,0.05^2),
]

[
\tau_\eta
\sim
{\rm HalfNormal}(0.03),
]

[
\mu_a
\sim
\mathcal N(0,0.5^2),
]

[
\tau_a
\sim
{\rm HalfNormal}(0.3),
]

[
\mu_Z
\sim
\mathcal N(0,1^2\ {\rm mag}^2),
]

[
\tau_Z
\sim
{\rm HalfNormal}(0.2\ {\rm mag}),
]

[
\sigma_{\rm star}
\sim
{\rm HalfNormal}(0.03\ {\rm mag}),
]

[
\sigma_{\rm extra}
\sim
{\rm HalfNormal}(0.05\ {\rm mag}).
]

These are only starting values. Put them in the configuration file and clearly state that they should be revisited after examining the posteriors.

For scale parameters, implement priors correctly in the log-posterior, including support restrictions and any parameter transformations or Jacobians that are required.

## Sampler

Use NumPyro's NUTS sampler with JAX 64-bit arithmetic. The potential function
uses the existing non-centered parameter vector, but evaluates the complete
wavelength-remapped passband and both trapezoidal photon-counting integrals at
every NUTS state. Do not substitute a passband emulator, Taylor expansion,
effective-wavelength approximation, or color-term likelihood.

Provide configurable settings for:

* number of independent chains;
* number of warmup steps;
* number of posterior samples per chain;
* thinning;
* random seed;
* chain execution method;
* target acceptance probability;
* maximum tree depth;
* diagonal or dense mass-matrix adaptation;
* progress display.

Use four chains by default. On CPU-only macOS, expose the requested number of
host JAX devices before sampling so `chain_method: parallel` runs one chain per
device without Python multiprocessing.

The code should:

1. construct reasonable initial values;
2. initialize each chain at a finite, lightly perturbed value;
3. reject invalid starting points;
4. run NUTS warmup and mass-matrix/step-size adaptation;
5. run posterior sampling;
6. save the full chain, log probability, and sampler extra fields;
7. save diagnostics in machine-readable JSON and CSV forms.

Report:

* mean acceptance probability per chain;
* divergent-transition counts per chain;
* split rank-normalized R-hat per sampled parameter;
* bulk effective sample size per sampled parameter;
* warnings for nonfinite diagnostics, R-hat above the configured reporting
  threshold, low effective sample size, or any divergence.

Do not silently claim convergence.

## Computational approach

It is acceptable to evaluate the synthetic-photometry integrals directly at every likelihood evaluation. The initial implementation does not need to be highly optimized.

However, avoid obvious repeated work:

* load and interpolate CALSPEC spectra once;
* interpolate the filter, aluminum², and CCD curves once; remap only the filter
  and multiply the fixed aluminum² and CCD response afterward;
* cache each star’s interpolated spectrum on the common wavelength grid;
* cache atmospheric transmission for repeated or rounded airmass values if convenient;
* vectorize over wavelength with NumPy;
* group observations by target, instrument, and airmass where useful.

Use trapezoidal numerical integration.

Keep the synthetic-photometry code modular so that a faster emulator or precomputed approximation can be added later.

## Software structure

Create a maintainable package rather than one monolithic script.

A reasonable structure is:

```text
passband_fit/
    __init__.py
    config.py
    io.py
    curves.py
    synthetic_photometry.py
    passband_model.py
    hierarchical_model.py
    sampler.py
    diagnostics.py
    plotting.py
    cli.py
tests/
    test_magnitude_conventions.py
    test_passband_remapping.py
    test_atmosphere.py
    test_hierarchical_parameters.py
    test_end_to_end_mock.py
example_config.yaml
README.md
pyproject.toml
```

Use:

* NumPy;
* SciPy;
* pandas;
* Astropy where useful for units and spectra;
* JAX and NumPyro;
* corner;
* matplotlib;
* PyYAML;
* compressed NumPy arrays or an equivalent format for chain storage.

Use type hints, docstrings, logging, and clear exceptions.

## Command-line interface

Provide a command such as:

```bash
python -m passband_fit.cli fit \
    --config example_config.yaml \
    --filter gp
```

Also provide commands or options for:

```bash
python -m passband_fit.cli validate ...
python -m passband_fit.cli plot ...
python -m passband_fit.cli summarize ...
```

The selected filter must be explicit.

The output directory should contain the filter name and a run timestamp or user-specified run label.

## Required output files

Save:

1. Copy of the resolved configuration.
2. Data-selection summary.
3. Mapping of integer indices to instruments and targets.
4. Nominal component curves.
5. Nominal combined passband.
6. MCMC chain.
7. Log-posterior chain.
8. Flattened posterior samples.
9. Parameter summary table.
10. Posterior covariance or correlation matrix.
11. Posterior median passband for each instrument.
12. Credible interval bands for each instrument’s passband.
13. Synthetic magnitudes under the original and fitted passbands.
14. Residual table for every observation.
15. Diagnostic plots.
16. Machine-readable JSON or YAML summary of the run.

## Parameter summary

For every scalar parameter, report:

* posterior median;
* 16th percentile;
* 84th percentile;
* posterior mean;
* posterior standard deviation;
* maximum-posterior sample value if available.

For each hierarchical family, report both:

* population hyperparameters;
* instrument-level parameters.

Also report the per-filter (\sigma_{\rm star}) hyperparameter and every physical
star offset (d_s).

For widths, report both:

[
\eta_j=\ln w_j
]

and:

[
w_j.
]

## Required plots

### 1. Original and fitted residuals

Define:

[
r_i^{\rm original}
==================

## m_{{\rm obs},i}

\left(
m_{{\rm syn},i}^{\rm original}
+
Z_{j[i]}^{\rm original}
\right),
]

and:

[
r_i^{\rm fit}
=============

## m_{{\rm obs},i}

\left(
m_{{\rm syn},i}^{\rm fit}
+
Z_{j[i]}^{\rm fit}
\right).
]

For the original model, estimate a zeropoint for each instrument using the nominal passband while holding all passband-shape parameters fixed. This creates a fair baseline comparison.

Plot:

* observed minus synthetic magnitude using the original passband versus original photon-weighted mean wavelength;
* observed minus synthetic magnitude using the fitted passband versus fitted photon-weighted mean wavelength.

Use a shared vertical scale when practical.

Distinguish telescope/camera systems by color or marker.

Also make versions distinguished by target.

Include:

* a horizontal line at zero;
* RMS;
* weighted RMS;
* median residual;
* robust scatter estimate such as MAD, for diagnostic reporting only;
* number of points.

### 2. Residuals versus other variables

Plot original and fitted residuals versus:

* airmass;
* observed magnitude;
* signal-to-noise ratio;
* telescope/camera system;
* target identifier.

These plots are important for deciding whether later versions need:

* site-dependent atmospheres;
* instrument-dependent extra scatter;
* night-level terms;
* robust likelihoods;
* additional time-, night-, or instrument-correlated residual structure.

### 3. Passband curves

Plot for each instrument:

* nominal passband;
* posterior median fitted passband;
* 68% credible interval;
* atmosphere-free instrumental passband;
* effective passband at one or more representative airmasses.

Also make a comparison plot showing all posterior median instrumental passbands together.

### 4. Population distributions

For each hierarchical parameter, show:

* instrument posterior estimates with credible intervals;
* inferred population mean;
* inferred population dispersion.

A forest plot is appropriate.

### 5. Corner plots

Create a corner plot for the selected filter including all sampled parameters, including:

* population means;
* population dispersions;
* all instrument shifts;
* all instrument widths when enabled;
* all instrument warps when enabled;
* all instrument zeropoints;
* per-star correlated-scatter hyperparameter;
* all per-star offsets;
* common extra-scatter parameter.

Because this can be high-dimensional, also create smaller corner plots for:

1. hyperparameters only;
2. each individual instrument’s passband parameters and zeropoint;
3. selected strongly correlated parameter groups.

Do not omit parameters from the full saved posterior merely because the full corner plot is visually crowded.

### 6. MCMC diagnostics

Plot:

* trace plots for all parameters;
* log-posterior trace;
* per-chain mean acceptance probabilities;
* R-hat and effective sample size;
* divergent transitions.

## Baseline nominal fit

Before running the full passband model, fit a baseline model in which:

* all passband-shape parameters are fixed at nominal values;
* each instrument zeropoint is fitted;
* the common extra-scatter parameter is fitted.

Save the baseline likelihood and residual metrics.

Compare the baseline and fitted model using at least:

* maximum log likelihood;
* posterior residual RMS;
* weighted residual RMS;
* AIC and BIC evaluated from the maximum-likelihood or maximum-posterior point, with clear documentation of the approximation.

Do not use these criteria as a substitute for posterior predictive checks.

## Posterior predictive checks

Generate posterior predictive residual realizations for a configurable subset of posterior samples.

Compare observed residual distributions with posterior predictive distributions for:

* overall scatter;
* residual mean;
* residual trend with photon-weighted mean wavelength;
* residual trend with airmass;
* residuals by instrument;
* residuals by target.

Include a simple binned residual trend versus photon-weighted mean wavelength, with uncertainties, for both original and fitted models.

## Mock-data validation

Create an end-to-end synthetic test.

The mock-data generator should:

1. load or construct a nominal passband;
2. create several synthetic stellar spectra covering a broad range of colors;
3. assign known hierarchical shifts, widths, warps, and zeropoints to multiple mock instruments;
4. generate observations at multiple airmasses;
5. add known measurement noise and common extra scatter;
6. fit the mock data;
7. verify that the injected values are recovered within reasonable posterior intervals.

Include cases for:

* shift-only;
* shift plus width;
* shift plus width plus warp;
* Vega magnitudes;
* AB magnitudes.

This test should explicitly catch:

* incorrect wavelength-shift sign;
* incorrect width convention;
* missing photon-counting factor of (\lambda);
* incorrect atmosphere exponentiation;
* incorrect AB unit conversion;
* incorrect zeropoint sign.

## Numerical and scientific checks

Add checks for:

* positive synthetic integrals;
* finite magnitudes;
* sufficient spectral coverage;
* passbands shifted entirely off the wavelength grid;
* extreme or invalid width values;
* overflows in the exponential warp;
* nonfinite posterior values;
* instruments with too few observations;
* targets represented by very few observations;
* invalid chain counts or nonfinite initial NUTS states.

Warn when the data provide little color leverage, such as when only a narrow range of stellar spectral types remains after cuts.

## Configuration example

Provide an extensively commented `example_config.yaml` containing sections for:

* selected magnitude system;
* selected passband model;
* input files;
* column names;
* wavelength units;
* common wavelength grid;
* pivot wavelength;
* warp wavelength scale;
* signal-to-noise cut;
* priors;
* sampler settings;
* plotting settings;
* output directory;
* atmosphere handling;
* chain execution method;
* random seed.

Example model section:

```yaml
model:
  magnitude_system: vega
  passband_mode: shift_width_warp
  pivot_wavelength_angstrom: 4800.0
  warp_scale_angstrom: 1200.0
  common_extra_scatter: true
  star_offsets: true
  robust_likelihood: false
  atmosphere_mode: fixed_kpno
```

Example selection section:

```yaml
selection:
  snr_min: 10.0
  require_positive_flux: true
```

## Documentation

Write a README explaining:

1. Scientific model.
2. Passband remapping convention.
3. Photon-counting synthetic magnitudes.
4. Vega and AB conventions.
5. Hierarchical priors.
6. Input-file formats.
7. Example commands.
8. Output products.
9. Diagnostic interpretation.
10. Known limitations.

Explicitly state the first-version assumptions:

* observations are conditionally independent given per-star offsets;
* one filter is fitted per run;
* one fixed KPNO atmosphere is used;
* atmospheric transmission scales as (T_1^X);
* one zero-mean, grey per-star offset hierarchy per filter;
* no night-level calibration terms;
* no site-dependent atmospheric parameters;
* no robust likelihood;
* one common extra-scatter parameter per filter;
* direct numerical integration is performed during likelihood evaluation;
* all instruments are partially pooled within the selected filter.

Also document likely future extensions:

* site-specific atmospheric curves;
* night-level zeropoints;
* correlated repeated observations;
* instrument-dependent extra scatter;
* wavelength-dependent CALSPEC spectral-shape nuisance terms;
* Student-(t) or mixture likelihood;
* atmospheric nuisance parameters;
* faster synthetic-photometry emulators.

## Development approach

Implement this in stages:

1. Input readers and validation.
2. Synthetic-photometry engine.
3. Nominal-passband residual calculation.
4. Passband remapping tests.
5. Non-hierarchical single-instrument fit.
6. Hierarchical multi-instrument fit.
7. MCMC persistence and restart support.
8. Diagnostic and science plots.
9. Mock-data recovery test.
10. Documentation and example configuration.

Before completing the implementation, run the unit tests and the mock end-to-end test. Fix any sign, unit, indexing, or parameterization errors revealed by those tests.

The final response should summarize:

* files created;
* model implemented;
* assumptions made;
* how to run the code;
* tests performed;
* any remaining scientific or computational concerns.

## Pre-implementation contracts

Freeze the following contracts before parallel implementation begins. Any later change to a public contract is made by the root/integrator and announced to every dependent agent.

### Runtime and package

* Use `/opt/anaconda3/bin/python` (currently Python 3.9.18) for development and verification.
* Support Python 3.9 or newer and use a `src/passband_fit/` package layout.
* Provide both a `passband-fit` console entry point and `python -m passband_fit`.
* Pin compatible dependency ranges in `pyproject.toml`; do not rely on undeclared packages.
* Use `float64` arrays internally.

### Configuration and input validation

* Use a strict, versioned configuration schema. Unknown keys are errors.
* `--filter` is required and overrides any filter value in the configuration.
* Resolve relative paths relative to the configuration file, not the process working directory.
* Load YAML with `yaml.safe_load`.
* Require explicit curve value semantics (`fraction`, `percent`, `extinction_mag_per_airmass`, or other supported type), wavelength unit, and wavelength convention (`air` or `vacuum`). Convert wavelengths to vacuum Angstroms and dimensionless throughput fractions internally.
* CALSPEC spectra use (S_lambda) in `erg s^-1 cm^-2 Angstrom^-1`; validate or explicitly convert other units.
* Sort wavelength tables, define duplicate-wavelength handling, and reject nonfinite wavelengths or values. Clip tiny negative physical throughput values to zero while reporting their count and minimum; fail for negative values beyond a configurable tolerance.
* Require finite positive flux uncertainty and finite positive airmass. Nonpositive flux and SNR failures remain counted selection rejections. Other malformed numeric rows fail validation rather than disappearing silently.
* Require measured flux to be an exposure-time-normalized count rate, or otherwise corrected to a common exposure/gain/aperture convention. Flux and uncertainty must have identical units.
* Preserve an immutable source-row identifier in all selected-data and residual outputs.
* If both an observation site and an instrument-to-site mapping are supplied, require agreement. The mapping is the source of truth when the observation table has no site column.

### Atmosphere file semantics

Support either unit-airmass transmission or extinction in magnitudes per airmass. Convert extinction coefficients using

\[
T_{\rm atm}(\lambda,1)=10^{-0.4 k(\lambda)}.
\]

Then apply (T_{\rm atm}(\lambda,X)=T_{\rm atm}(\lambda,1)^X). The supplied
atmospheric table must reach negligible transmission at its blue endpoint, and
the CCD QE must reach negligible response at its red endpoint, before zero fill
is allowed. Grid and spectral coverage use the complete transformed-filter ×
fixed-aluminum² × fixed-QE × atmosphere response, rather than requiring each
component to span the atmosphere-free warped-filter envelope. Sentinel values
must be handled explicitly in configuration or preprocessing and must never be
mistaken for transmission fractions.

Use the same observation-specific atmosphere in both the source numerator and the Vega or AB reference denominator. Do not round airmass by default. If optional airmass grouping is enabled, record the rounding precision in the resolved configuration and manifest.

### Grid and spectral coverage

* Define non-negligible response by a configurable threshold relative to the maximum effective response, initially `1e-6`.
* Pad the common wavelength grid until the complete response at every configured
  shift/width/warp corner is negligible at both endpoints. The atmosphere sets
  the physical blue edge and the fixed CCD QE sets the red edge. Priors may be
  broad, but numerical hard bounds are explicit and proposals outside them
  return negative infinity.
* Require each target spectrum and the Vega reference spectrum to cover the full non-negligible response region. Fail validation when coverage is insufficient and report every affected spectrum plus the required and available wavelength intervals. The analytic AB reference has no finite-coverage limitation.
* Reject a posterior proposal if non-negligible response reaches a grid endpoint, the passband is shifted off the grid, or material integrated area would be truncated.
* Add a grid-convergence test in which halving the spacing changes nominal synthetic magnitudes by less than a configurable tolerance. Start with `0.0005 mag`; revisit this tolerance against the best observational uncertainty.
* Compute residual-plot mean wavelengths per observation, including its airmass. Label all atmosphere-free and atmosphere-specific wavelength/passband products explicitly.

### Likelihood, parameters, and summaries

* Include the full heteroscedastic Gaussian log likelihood, including its normalization. Analytically marginalize the zero-mean Gaussian shared star offsets with diagonal-plus-rank-one same-star blocks, and reconstruct their exact conditional posterior for reported physical products. Verify against dense covariance calculations in tests.
* A single stable `ParameterLayout` owns sampled names, ordering, packing, unpacking, and derived physical parameters for all three model modes.
* Sample positive scale parameters in explicitly documented coordinates. When sampling their logarithms, include the required change-of-variable Jacobian for priors defined on the physical scale. Invalid states return negative infinity, never NaN.
* Do not instantiate instrument-level parameters for instruments with zero selected observations. Report rows and distinct CALSPEC targets per instrument, warn on weak color leverage, and label evidently prior-dominated parameters.
* Document that positive warp (a_j) suppresses the red side relative to the blue side.
* Fitted residuals use the posterior median of each observation's complete prediction, (m_{\rm syn,i}+Z_j+d_s), evaluated over posterior draws. Do not substitute a pointwise-median passband for a parameter draw.
* Passband credible bands are percentiles of curves computed from posterior draws. Normalization for plotting is allowed but must not alter curves used for photometry.
* The baseline model is fitted by maximum likelihood for instrument zeropoints and common extra scatter. AIC and BIC use maximum data log likelihood, never log posterior; define (n) as used observations and (k) as all fitted free coordinates, and label hierarchical AIC/BIC as exploratory.

### Persistence and CLI semantics

* `validate --config CONFIG --filter FILTER` fully loads, converts, selects, and prepares data without sampling.
* `fit --config CONFIG --filter FILTER` creates a new nonempty-safe run directory. NumPyro runs are atomic at the run level; legacy emcee `--resume` directories are rejected explicitly rather than mixed with NumPyro output.
* `plot --run RUN_DIR` and `summarize --run RUN_DIR` use persisted artifacts without rerunning inference or depending on the original working directory.
* Use compressed NumPy arrays as the authoritative unthinned chain, log-probability, and NUTS-extra-field store. Thinning applies only to derived flattened exports.
* Persist parameter names/order, dimensions, chain count, warmup/sampling iterations, chain method, schema/package versions, resolved-config hash, input SHA256 hashes, seed, and run status in a manifest.
* Handle interruption cleanly, write completed artifacts atomically where applicable, sanitize user-supplied path components, refuse accidental overwrite, and return a nonzero CLI status on validation or runtime failure.

### Test tiers and acceptance

1. Fast deterministic tests cover configuration, selection accounting, units, sorting/duplicates, scale invariance, Vega self-zero, AB flat-(f_nu) zero, the photon-counting lambda factor, combined shift/width sign, warp sign, atmosphere conversion/exponentiation, coverage/grid edges, parameter layout/Jacobians, and likelihood at injected truth.
2. CLI integration tests use a tiny fixed mock to cover validation, fit, summarize, plot, output manifest completeness, no-overwrite behavior, and rejection of legacy resume artifacts.
3. Slow tests, marked `slow`, perform fixed-seed stochastic recovery with declared 95-percent-interval or standardized prediction-error criteria. Cover all model modes and both magnitude systems across the suite without requiring six production-length chains in every default test run.
4. Save unthinned production samples. Report per-parameter R-hat and effective sample size plus divergences; never promote a smoke run as evidence of convergence.
5. The final gate runs `/opt/anaconda3/bin/python -m pytest`, the marked slow suite, CLI help, validation of each configured filter, a baseline NumPyro smoke fit, and artifact-manifest verification.

## Multi-agent implementation workflow

Use the four available active slots as one root/integrator plus three implementation agents. All agents share one filesystem, so isolation is enforced by file ownership rather than branches. No two agents edit the same file concurrently. An ownership transfer requires an explicit root message.

### NumPyro conversion workflow (current)

* Root/integrator owns `src/passband_fit/numpyro_model.py`, pipeline/CLI/output
  integration, legacy-emcee removal, documentation, persisted artifact schemas,
  and end-to-end real-input verification.
* `science_review` owns the new exact full-grid
  `src/passband_fit/jax_photometry.py` kernel and NumPy/JAX parity and gradient
  tests. Its kernel must retain wavelength remapping and both photon integrals;
  it may not introduce a surrogate likelihood.
* `workflow_review` owns the standalone `src/passband_fit/numpyro_sampler.py`
  NUTS adapter and focused tests for chain shapes, seeds, host-device chain
  execution, log probability, R-hat, ESS, acceptance, and divergences.
* `software_review` owns the strict NumPyro sampler configuration schema,
  dependency metadata, and migration of every filter YAML while preserving all
  nonsampler scientific settings.

Integration gates are: (1) focused agent tests; (2) exact JAX/NumPy log-posterior
parity on real prepared data; (3) the full fast suite; (4) a fresh-process
real-data smoke run showing the requested and effective parallel chain method,
multiple CPU devices, a saved chain, and saved R-hat/ESS diagnostics. Root alone
resolves cross-file interfaces and performs the final gates.

### Ownership

* Root/integrator: `Plan.md`, `pyproject.toml`, `README.md`, `src/passband_fit/__init__.py`, `src/passband_fit/__main__.py`, `src/passband_fit/cli.py`, `src/passband_fit/pipeline.py`, `src/passband_fit/mock_data.py`, `tests/test_end_to_end_mock.py`, public domain types/interfaces, integration, and final test commands.
* `data-config` agent: `src/passband_fit/config.py`, `src/passband_fit/io.py`, `src/passband_fit/output.py`, `example_config.yaml`, `tests/test_config.py`, and `tests/test_io.py`.
* `photometry-kernel` agent: `src/passband_fit/curves.py`, `src/passband_fit/synthetic_photometry.py`, `src/passband_fit/passband_model.py`, `tests/test_curves.py`, `tests/test_magnitude_conventions.py`, `tests/test_passband_remapping.py`, and `tests/test_atmosphere.py`.
* `inference-diagnostics` agent: `src/passband_fit/hierarchical_model.py`, `src/passband_fit/sampler.py`, `src/passband_fit/diagnostics.py`, `src/passband_fit/plotting.py`, `tests/test_hierarchical_parameters.py`, `tests/test_sampler.py`, and `tests/test_diagnostics.py`.

Agents may read any file but do not opportunistically edit another owner's files. Root alone changes package exports, public contracts, CLI orchestration, metadata, plan, documentation, and end-to-end tests. Do not run a broad formatter while other agents are active.

### Phase 0: contract freeze

Root defines the typed configuration, `PreparedData` and curve/spectrum array shapes, observation index conventions, callable interfaces from loaders through posterior evaluation, `ParameterLayout`, exception policy, artifact filenames/schema, and an immutable tiny mock fixture.

Gate 0: the package imports, the example configuration validates structurally, public signatures and schemas are reviewed, and every unresolved real-input issue is either answered or represented by an explicit configuration mapping.

### Phase 1: parallel foundations

* `data-config` implements strict configuration, input readers, selection accounting, unit conversion, mappings, and FITS/text spectrum support.
* `photometry-kernel` implements interpolation, coverage checks, atmosphere conversion, Vega/AB photon-counting photometry, passband remap/warp, and photon-weighted mean wavelength.
* `inference-diagnostics` implements the deterministic parameter layout, non-centered transforms, prior support/Jacobians, and a likelihood consuming an injected photometry evaluator with no CLI coupling.
* Root scaffolds packaging, shared contracts, and integration fixtures.

Gate 1: all focused deterministic tests pass, especially AB units, Vega zero, scale invariance, Gaussian remap signs with simultaneous width and shift, atmosphere conversion/exponentiation, selection counts, prior support/Jacobians, and mode-dependent dimensions.

### Phase 2: vertical slice and persistence

* `data-config` adds resolved-config, index, selection-summary, manifest, and output writers and validates the supplied inputs.
* `photometry-kernel` adds precomputed trapezoid weights, cached spectra/reference arrays, exact unique-airmass atmosphere matrices, grouped repeated evaluations, nominal-model calculations, and numerical guards. A likelihood call performs no file I/O.
* `inference-diagnostics` adds NumPyro NUTS execution, multi-chain initialization, acceptance/divergence/R-hat/ESS reporting, compressed-chain persistence, and deterministic tests. CPU chain parallelism uses JAX host devices rather than Python multiprocessing.
* Root connects `validate`, baseline nominal fit, and a tiny in-memory fit through the pipeline and CLI.

Gate 2: real-input validation reaches only known documented blockers; nominal residuals are finite for valid fixtures; a short fixed-seed multi-chain run writes and reopens without parameter-order drift; JAX and NumPy likelihood values agree at fixed states; invalid configurations fail with clear messages.

### Phase 3: science outputs and recovery

* `inference-diagnostics` adds summaries, covariance/correlation, exploratory AIC/BIC, posterior predictive checks, and all required plots.
* Root completes the mock generator, CLI commands, output manifest, baseline/full-model comparison, and end-to-end tests.
* `data-config` audits persisted outputs and resolved input provenance.
* `photometry-kernel` performs independent unit/sign review, grid-convergence checks, prior-predictive passband checks, and profiling, editing only owned files.

Gate 3: quick tests exercise all modes and magnitude systems deterministically; at least one small MCMC recovery test verifies injected predictions/parameters under declared tolerances; the complete artifact manifest is produced and readable by plot/summarize commands.

### Phase 4: stabilization and handoff

The three agents perform read-only cross-reviews for data/config/output, photometry/signs/units, and inference/restart/diagnostics. They report findings to root; root integrates cross-cutting fixes, finalizes documentation, and runs the full test gates. Do not begin production inference on real observations until validation passes and the user-selected filter/input mappings are confirmed.

Every agent handoff states exact files changed, public APIs added or changed, test commands/results, assumptions, and open risks. A gate failure returns to the owning agent. Interface changes after Gate 0 require root approval and a broadcast to all dependent agents.

This directory is not currently a Git repository. Unless Git initialization is authorized, shared-file ownership and root-maintained handoff manifests replace branch/cherry-pick integration. If Git is initialized later, root creates phase-boundary commits while file ownership remains in force.

## Current workspace audit and resolved input decisions

### I-band divergence-reduction workflow (2026-07-19)

The active integration pass keeps the exact full-grid photon-counting
likelihood and the existing data-informed, nearly common chain initialization.
It makes these coordinated changes before another production run:

1. Marginalize each zero-mean Gaussian per-star residual analytically, sample
   only `sigma_star`, and reconstruct physical star-offset draws from their
   exact conditional posterior for residuals, summaries, forests, and posterior
   predictive checks. Check NumPy and JAX likelihood parity against a dense
   covariance reference.
2. For `I.yaml` and `I_2.yaml`, use a 2700--15000 Angstrom grid, a numerical
   shift guard of +/-1000 Angstrom, `Normal(0, 300 Angstrom)` for the population
   mean shift, `HalfNormal(100 Angstrom)` for its population scale, and width
   guards 0.6--1.4. These wide guards remain numerical safety limits, not the
   scientific prior.
3. Use 2000 warmup draws, `target_accept_prob: 0.95`, and a dense mass matrix.
   Persist posterior proximity to all numerical parameter bounds and wavelength
   endpoints in `boundary_diagnostics.json`.
4. Reject truncated atmosphere and CCD-QE inputs before interpolation can hide
   them with zero fill. The I `shift_width` envelope requires red support beyond
   12400 Angstrom; the subsequently extended 14400 Angstrom atmosphere and
   14000 Angstrom QE now satisfy validation.
5. Run focused analytic/JAX tests, the full test suite, both I validations, and
   only then a fresh smoke run. Production inference starts only after input
   support validation passes.

The initial audit on 2026-07-15 found Python 3.9.18 and all named runtime/test dependencies installed. The user subsequently cleaned the observation table and supplied the missing Vega spectrum. Use these resolved input rules:

* Parse the telescope/camera identifier from the first portion of each image filename, for example `lsc1m004-fa03`. Derive the site code from its leading three characters, for example `lsc`.
* Use `CALSPEC_spectra/alpha_lyr_stis_012.fits` as the Vega reference.
* Interpret `passband_ingredients/kpnoextinct.dat.txt` as extinction coefficients in magnitudes per airmass and convert them to transmission before exponentiation by airmass.
* Interpret `passband_ingredients/aluminum_squared.txt` as already containing (A(\lambda)^2) and use it once.
* Interpret nominal-filter wavelengths as nanometers, `QE.txt` values as percent, and all CALSPEC wavelengths as vacuum Angstroms. Work internally in vacuum Angstroms.
* Use a starting grid-convergence tolerance of 0.0005 mag.
* For the supplied digitized filter curves, use a relative response-support threshold of `1e-3`, above their measured approximately `3e-4` signed digitization-noise floor. This is a dataset-specific override of the earlier generic `1e-6` starting value.
* The cleaned observation table uses `zs` consistently and maps Feige 22 consistently.
* Count and exclude the two rows with zero flux uncertainty; do not abort other valid observations.

Treat `PSF_residual_RMS` as an auxiliary diagnostic column unless the user specifies that it contributes to measurement uncertainty. Do not silently infer any of the other mappings above.
