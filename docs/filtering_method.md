# Magnetic waveform filtering and spatial measurements

The analysis separates magnetic variability into fine-scale, intermediate-scale,
large-scale and mean-retaining trend families for Solar Orbiter, ACE, Wind, DSCOVR,
Solar-1 and ARTEMIS-P1/P2. These are spatial-scale waveform components, not a
partition into turbulence regimes. All families use the same prepared magnetic
record and convected-distance coordinate.

The filtering context is 17 January 2026 00:00 to 23 January 2026 00:00 UTC.
The retained record is 18 January 00:00 to 22 January 00:00 UTC, with the stop
excluded: 5,760 one-minute rows per spacecraft and family. Preparation uses
available observations within the context; it does not synthesize observations
before the first native sample. Solar Orbiter vectors remain in RTN. Near-Earth
magnetic components use GSM: native provider products except for definitive
DSCOVR GSE vectors rotated at their native UTC epochs with SpacePy CTrans and
supplied IGRF14 coefficients. Rotation precedes cadence preparation and filtering.
The original scalar channels, timestamps and quality policies are retained;
DSCOVR remains the same definitive CDAWeb source. Ephemerides and plasma velocity
components remain GSE. Reference files and their manifest are described in the
[observation preparation instructions](../README.md#observation-preparation).

## Magnetic preparation and speed coordinate

After mission quality screening, magnetic data are prepared on a one-second
grid as follows. The grid spacing does not imply independent one-second
observations for every instrument.

| Spacecraft | One-second preparation |
|---|---|
| Solar Orbiter | Arithmetic means of the native 8 Hz observations, separately for components and scalar magnitude |
| ACE, DSCOVR, Solar-1 | Arithmetic means of the approximately one-second observations |
| Wind | Arithmetic means in one-second bins; forward hold for at most two empty bins |
| ARTEMIS-P1/P2 | Arithmetic means of GSM components and retained native-GSE scalar norm in one-second bins; hold each channel for at most three empty bins |

The native ARTEMIS survey cadences are measured from the supplied timestamps.
The remaining missing bins are preserved in the prepared record. Magnetic
components and scalar magnitude are stored separately; filtering scalar
magnitude is not equivalent to taking the magnitude of a filtered vector.
ARTEMIS scalar strength is formed from the original native GSE vector before
binning and holding; it is not reconstructed from held GSM components.

The speed profiles use one-minute plasma records. Solar Orbiter, Wind and
Solar-1 use their own proton speeds, with linear interpolation limited to 30
missing minute rows in each direction. ACE uses Solar-1 speed shifted by the
measured shock-arrival difference. DSCOVR uses its own speed before its measured
shock, with the same bounded local interpolation and the shifted Solar-1 profile
filling remaining missing values, and the proxy afterward. ARTEMIS uses its own merged ESA speed when it lies strictly between
100 and 3,000 km/s and differs by at most 30% from an available shifted Solar-1
proxy; otherwise it uses the proxy. Own ARTEMIS speeds are retained when the
proxy is unavailable. Shifted profiles are sampled by nearest timestamp within
90 seconds. Local-profile interpolation and these source choices are recorded
by [build_speed_profiles.py](../python/pipeline/build_speed_profiles.py).

For the spatial mapping, interpolate each profile linearly in UTC to the
prepared magnetic timestamps, using the nearest finite endpoint at context
edges. Remaining speed gaps longer than 30 minutes are rejected except for the
specified Solar Orbiter interval [19 January 04:23, 08:00) UTC, which is linearly
bridged. Integrate the prepared speed by the trapezoidal rule:

\[
 x(t)=\frac{1}{R_E}\int_{t_0}^{t}V(t')\,dt',\qquad R_E=6371\ \mathrm{km}.
\]

This changes the coordinate; magnetic values remain in nT. Linearly interpolate
each magnetic component and scalar magnitude onto a uniform grid with spacing
0.04 R_E. That spacing must not exceed the smallest distance step between
prepared one-second samples. Reference spectra and waveform filters use this
same mapping and spatial magnetic record.

## Reference scales and filter bands

Four reference lengths are calculated from ACE, Wind, DSCOVR and Solar-1:

| Reference | Definition |
|---|---|
| Short | Median measured shock timing uncertainty multiplied by the median prepared speed at the four respective shock times, divided by R_E |
| Intermediate spectral reference | Median dominant sheath wavelength from the four unfiltered scalar-magnitude spectra |
| Sheath | Median of x(sheath end) − x(shock) |
| Ejecta | Median of x(ejecta end) − x(sheath end) |

The short reference is the product of two medians. Each speed is evaluated at
its spacecraft's exact shock t50, including fractional seconds. Boundary
coordinates are linearly interpolated in the prepared x(t), and extents are
differences of those coordinates.

For each sheath spectrum, select the unfiltered spatial samples from the
measured shock, inclusive, to the sheath end, exclusive. Subtract a linear
trend and calculate a single full-sheath Hann-window spectrum using SciPy
`welch`, with constant detrending and density normalization. The segment is the
whole selected record, so no averaging across multiple segments occurs. At
least 32 finite prepared time samples and 32 spatial samples are required.
Choose the strongest positive spectral ordinate whose wavelength is shorter
than the sampled sheath span. Missing prepared magnetic bins are bridged by
the spatial interpolation; their counts are saved with the spectral result.

Let the ordered reference lengths be L_short, L_intermediate, L_sheath and
L_ejecta. The three internal wavelength boundaries are the geometric means of
adjacent references. The lower boundary is

\[
 \lambda_0=2.5\,\max(\Delta t_{P1},\Delta t_{P2})
             \frac{1100\ \mathrm{km\,s^{-1}}}{R_E},
\]

where the ARTEMIS cadences are measured medians. The multiplier 2.5 and reference
speed 1100 km/s are chosen sampling-floor parameters. They are separate from
the measured speeds used for x(t) and the short reference.

| Family | Wavelength range |
|---|---|
| Fine-scale | λ0 to √(L_short L_intermediate) |
| Intermediate-scale | √(L_short L_intermediate) to √(L_intermediate L_sheath) |
| Large-scale | √(L_intermediate L_sheath) to √(L_sheath L_ejecta) |
| Trend | Above √(L_sheath L_ejecta), retaining the mean |

The references must be positive and strictly ordered, and the sampling floor
must leave a resolved fine-scale band. Method parameters are defined in
[waveform_bands.py](../python/config/waveform_bands.py); measured references,
cadences and edges are written to `processed/waveform_bands.json` and
`processed/waveform_scales.csv`.

Publication names are shared through `WAVEFORM_FAMILY_LABELS`. The product
identifiers `discontinuity`, `substructure`, `sheath` and `trend` remain stable
in saved interfaces. The intermediate spectral reference likewise retains its
internal `substructure` key. Physical sheath and ejecta reference extents keep
their region-based definitions.

## Filtering, gaps and minute reduction

Each component and scalar magnitude is filtered with a fourth-order Butterworth
design applied forward and backward in second-order sections (`sosfiltfilt`).
The three band-pass outputs are signed residuals. The trend is a low-pass output
that retains the slowly varying mean field.

Magnetic gaps are linearly bridged on the spatial grid for the filtering
operation. After interpolation back to the one-second timestamps, samples
missing in the prepared input are restored as missing. The bounded Wind and
ARTEMIS holds remain part of that input.

For each band, let λ_min be its lower wavelength boundary and λ_long its upper
boundary, or its cutoff for the trend. A gap is considered long when the
separation of its bounding finite samples exceeds max(25 R_E, 0.25 λ_min).
Such gaps are masked together with 0.5 λ_long on either side. Record edges are
masked over 1.5 λ_long from each end. Masks are calculated separately for each
magnetic channel. The implementation is in
[structure_bands.py](../python/analysis/structure_bands.py).

Only finite, valid filtered values enter each left-closed UTC minute. Store the
arithmetic mean, sample standard deviation (`ddof=1`) and finite-grid count.
One value supports a mean; at least two support a standard deviation. The
standard deviations describe within-minute variability, not measurement
uncertainty, standard error or confidence intervals. Counts describe the
prepared grid, not independent instrument observations.

Wind's bounded, isolated missing minutes are identified in its observed minute
product, with a maximum gap of one minute. At those flagged timestamps only,
linearly interpolate filtered minute means and spreads in UTC. Retain zero
sample counts, `observed_valid=False` and `source_interpolated=True`. These
flags remain available for exclusion and sensitivity calculations.

## Intervals and spatial metrics

Figure 4 evaluates every family, including the mean-retaining trend, over a
spacecraft-specific interval from
[build_waveform_intervals.py](../python/pipeline/build_waveform_intervals.py).
The search window runs from the minute containing the spacecraft's shock minus
two hours to the minute containing its ejecta end plus two hours. Inside it, the
interval runs from the minute where the cumulative sum of the squared saved
one-minute `bt_f` reaches 5% to the minute where it reaches 95%; two hours are
then added on each side. For the trend, the squared signal is essentially
|B|². The cumulative-energy fractions limit the influence of low-energy tails.
The fractions, search margin and padding are method settings in
[waveform_bands.py](../python/config/waveform_bands.py);
the intervals are recalculated each run and shown in Supplementary Figure 1.

Metric intervals are selected in UTC. Their distance coordinates are samples of
the same prepared x(t), with the exact local shock t50 set to zero; the anchor
sets only the distance origin. Inverse UTC values are
interpolated from the saved minute and exact-anchor coordinates.

For scalar residual q and vector residual b, the reported RMS values are
√(∫q² dx / ∫dx) and √(∫|b|² dx / ∫dx), using the same all-component valid mask.
Here |b|² = bx² + by² + bz² for the filtered components. RMS uses these values
directly without subtracting their interval means.
Following the variance-based magnetic-compressibility definition cited by
[Wang et al. (2020)](https://doi.org/10.3847/1538-4357/abb883) and
[Wang et al. (2023)](https://doi.org/10.5194/angeo-41-129-2023),
the third row of Figure 4 reports the centered scalar-to-vector variance ratio
`scalar_to_vector_variance_ratio`. For each saved one-minute channel a, define
its spatially weighted mean and variance on the full selected support S as

\[
 \langle a\rangle_S=\frac{\int_S a\,dx}{\int_S dx},\qquad
 \mathrm{Var}_S(a)=\frac{\int_S(a-\langle a\rangle_S)^2\,dx}{\int_S dx}.
\]

The ratio is Var_S(q) / [Var_S(bx) + Var_S(by) + Var_S(bz)]. Each channel is
centered on its own mean over the same interval, with the same all-component
valid mask and spatial weights. It measures variance of the saved minute means;
within-minute sample spreads do not enter it. Spatial integrals use connected
valid segments, with a 25-R_E maximum gap separation.

Central 5–95% scalar-power widths describe extent inside the selected interval;
they do not crop the RMS or variance-ratio calculations. Trend RMS and width
retain the mean field, while the variance ratio centers each low-pass channel
over the selected interval.

An orthogonal rotation preserves each unfiltered vector's norm. A fixed rotation
also preserves vector RMS and summed centered component variances. The GSE--GSM
rotation varies with UTC, however, and does not generally commute with filtering
or interval-mean subtraction. Filtered vector metrics and directional boundary
diagnostics are therefore recalculated rather than assumed invariant. RTN and
GSM component signs are not directly compared. Additional sensitivities exclude
flagged interpolated minutes and
replace linear speed-gap interpolation with previous-value and next-value
holds. These are definition sensitivities, not statistical uncertainty bounds.

## Reproducible products

The [publication runner](../python/pipeline/run_publication_pipeline.py)
recalculates boundaries, speeds, reference scales, filters, intervals and metrics
in dependency order from the supplied observations. Missing or invalid
prerequisites fail explicitly. Scale and coordinate readers check input
fingerprints; downstream readers validate product schemas and scientific
consistency.

`processed/bands/<craft>__<family>.csv` contains `bx_f`, `by_f`, `bz_f`, `bt_f`,
within-minute `*_std_f`, `*_filtered_grid_samples`, and the flags `observed_valid`,
`valid`, `std_valid` and `source_interpolated`. Companion metadata records the
measured filter ranges, processing settings, input files and gap treatment.
Shared distance samples are stored under `processed/spatial_coordinates/`, with
anchor and input fingerprints in `processed/spatial_coordinates.json`.

`results/waveform_intervals.json` records each spacecraft's and family's
search window, energy edges and padded interval. The chain metric CSV/JSON
products retain waveform measurements, eligibility decisions and
interpolation/speed sensitivities. These products contain
the measured values used by the figures; no derived scale, boundary or lag is a
fixed source-code parameter.
