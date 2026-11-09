# Event-specific magnetic wavelet coherence

Figure 5 compares three complementary measurements. The first row retains the
existing peak lagged Pearson correlation of four-day, one-minute filtered scalar
waveforms. The second row measures local magnitude-squared wavelet coherence of
the original prepared field strength within each spatial band. These values are
not the squares of the first-row correlations. The third row gives the Pearson
correlation of filtered GSM Bz at each pair's scalar-derived lag.

The three columns show the band-pass families; the trend is omitted. Every panel
displays the 15 pairs among the four L1 spacecraft and ARTEMIS-P1/P2. Separate
exponential fits use the same ten incident links: six L1--L1 and four L1--P1.
Five P2 links provide context. Solar Orbiter remains in the original scalar
calculations and seven-observer common wavelet support but is omitted from the
near-Earth display. L_cor, L_coh and L_cor,z label the respective fitted 1/e
lengths. Fit support and extrapolation qualifications remain in the generated
caption and numerical records.

## Input, alignment and common support

`pipeline/build_wavelet_coherence.py` recomputes the analysis from the current
prepared one-second magnetic observations, measured shock times, speed profiles
and derived waveform bands. It uses the same 0.04-R_E spatial mapping as the
reference spectra and waveform filters. No saved experimental maps or historical
boundary times, speeds, lags or band edges substitute for those prerequisites.

Each spacecraft's own measured shock is set to distance zero. This registers
the event before comparing local spatial variations; no additional lag is
selected to maximize coherence. The six-day context supplies padding around
the common four-day science interval. Coherence is calculated from the
unfiltered scalar field strength, not from an already band-pass-filtered series.

The transform uses a Morlet wavelet with nondimensional frequency 6 and twelve
scales per octave. Anti-aliased downsampling retains at least twelve samples
per shortest wavelength. Transform scales extend beyond each reporting band
to support scale smoothing. Reporting uses the band edges measured in the
current run. The trend comparison has a finite observable upper wavelength
determined by the available context and common science extent; it excludes DC.

Routine cadence holes are interpolated under the existing spatial-gap policy.
Material gaps and record ends are excluded using the wavelet cone of influence.
Within each family, every pair uses the same positions and wavelengths: all
seven observers and every retained wavelength must be valid. Thus variations
in pair-specific edge coverage do not become a distance trend.

## Coherence and activity weighting

For wavelet coefficients W_i and W_j at scale s, the smoothed local
magnitude-squared coherence is

\[
 R^2_{ij}=\frac{|S(W_i W_j^*/s)|^2}
 {S(|W_i|^2/s)S(|W_j|^2/s)}.
\]

The smoothing follows PyCWT's Morlet implementation. The row-wise calculation
is checked against PyCWT on synthetic signals. Coherence describes consistent
local cross-spectral relationships; it does not require equal magnetic
amplitudes, identify a wave mode, or establish causality.

The common activity weight at each position and wavelength is the mean of the
four L1 smoothed auto-powers. If a transform is standardized, its physical
auto-power is restored by multiplying by that record's magnetic variance.
Every pair receives this same observed weight. The reported summary is
sum(w R²)/sum(w), integrated over common distance samples and logarithmically
spaced wavelengths. This emphasizes wavelengths and locations carrying the
event's field-strength variability. It does not select weights from coherence
itself and is not a fraction of vector magnetic energy.

## Benchmarks, uncertainty and distance fits

Twenty independent white-noise pairs per family are transformed on the same
grid and summarized with the fixed observed weights and support. Their mean
and 95th percentile are descriptive benchmarks. They are not calibrated
significance thresholds for solar-wind noise or instrument errors.

The uncertainty calculation resamples synchronized, non-circular spatial
blocks of the precomputed pair numerators and common weight denominator.
All pairs receive the same drawn positions. Each draw recomputes the ratio of
summed numerator to summed denominator, retaining the activity weighting.
The primary calculation uses 1,000 realizations. Block length is at least the
longest retained wavelength and includes the measured autocorrelation lengths
of the weighted coherence profiles, their numerators and the denominator.
Half- and double-block calculations assess sensitivity where feasible.

The 2.5th–97.5th percentile ranges are conditional on the observed transforms,
weights, alignment, support and block prescription. They are not a full
calibrated confidence interval for a population of CMEs. Activity concentration
is recorded as a diagnostic, not used as a gate to suppress ranges. When the
trend spans fewer than one long-wave block, only descriptive point values are
reported: there is no informative trend interval or fitted trend length.

For the same ten near-Earth links used by the correlation fit (six L1–L1 and
four L1–ARTEMIS-P1), the primary descriptive curve is R²(d)=exp(-d/L_coh).
Here L_coh is the fitted distance at which the wavelet summary reaches 1/e.
The correlation-row length L_cor instead refers to Pearson r reaching 1/e;
the two lengths describe different estimators and are not interchangeable.
RMSE and MAE measure its residuals. Fit support, bootstrap behavior and the
sampled distance range determine whether a distance is supported, unresolved
or beyond the observed baselines; high coherence alone does not resolve a
decay length. Support requires at least three fitted links above the white
benchmark, including both link groups, at least 95% finite resampled summaries,
and a decreasing coherence–distance slope in at least 95% of those summaries.
These are descriptive fit-support checks, not a white-noise hypothesis test.
A length is classified within, above or below the sampled range only when at
least 95% of conditional length estimates lie in that region; otherwise it is
unresolved. A separate white-floor fit is saved as sensitivity, without
replacing the primary curve. Infinite or unsupported endpoints have explicit
status fields rather than substituted finite values.

These are measurements of this event. The baselines mix radial and transverse
separations, so the fitted distances do not isolate radial evolution or imply
an intrinsic isotropic plasma length. Correlation and wavelet coherence use
the same observations and provide complementary evidence, not independent
replications.

## GSM Bz within the scalar-aligned structures

`pipeline/build_bz_persistence.py` reads the current filtered one-minute GSM
components for the six near-Earth spacecraft and all three band-pass families.
It verifies the frame in both the observer registry and prepared-source metadata.
For each of the 15 pairs, it calculates centered Pearson correlation of Bz at
the peak scalar lag already measured from that pair's field-strength waveforms.
It does not independently select a Bz lag. Thus it asks whether the same
scalar-aligned magnetic structures retain similar Bz variations.

Uncertainty uses 1,000 synchronized, non-circular moving-block realizations.
Blocks resample source-time origins on the original clock. Each realization
repeats the scalar lag search, then compares original samples A[j] and B[j+lag];
it does not create artificial pairs across concatenated block boundaries.
The interval means are recalculated in each draw. The block duration is the
larger of the current scalar dependence prescription and the longest measured
autocorrelation duration among Bz and Bz squared across the six observers.
Pair bars and fit bands give 2.5th--97.5th percentile bootstrap intervals.
They are conditional on the prepared observations, coordinate realization,
filters, measured boundaries and fixed reference geometry; calibration and
boundary uncertainties are not propagated.

Each family has a new equal-link least-squares fit r_z(d)=exp(-d/L_cor,z), using
the same ten incident links as the scalar fits. Signed Bz correlations are
retained, including negative values; the exponential model is a descriptive
summary and cannot represent an anticorrelated branch. RMSE and MAE report
its residuals. Scalar null-test acceptance criteria are not transferred to Bz.
A fitted length beyond the sampled separations remains an extrapolation.
High centered Bz correlation alone establishes similar variations, not equal
amplitudes or identical mean polarity.

## Products and reproducibility

The publication runner executes this stage after the original correlation
inference and before Figure 5. Products under `python/output/results/` include
`wavelet_coherence.json`, `wavelet_coherence.csv`,
`wavelet_coherence_persistence_fits.csv` and `wavelet_coherence_draws.npz`.
The separate Bz stage writes `bz_persistence.json`, `bz_persistence.csv`,
`bz_persistence_fits.csv` and `bz_persistence_draws.npz` before Figure 5.
Metadata records inputs, method settings, support and uncertainty definitions.
Transform intermediates are runtime caches under `python/output/.cache/` and
are regenerated by the build. Publication and diagnostic figures are PDF only.

Method references: [PyCWT reference](https://pycwt.readthedocs.io/en/development/reference/)
and [Grinsted et al. (2004)](https://doi.org/10.5194/npg-11-561-2004).
