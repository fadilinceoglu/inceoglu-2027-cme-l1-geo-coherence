# Multispacecraft tracking of the 19 January 2026 ICME

Analysis code for **Scale-dependent magnetic coherence and geosynchronous
responses to the January 2026 coronal mass ejection**. The study tracks magnetic
waveforms from Solar Orbiter through ACE, Wind, DSCOVR and Solar-1 to
ARTEMIS-P1/P2, and compares them with the GOES-18/19 magnetic response.

The analysis starts from mission observations and produces event boundaries,
spatially filtered waveforms, physical metrics, pairwise correlation inference,
activity-weighted wavelet coherence, GOES reconstructions and publication figures. The retained waveform record is
18–21 January 2026; filtering uses 17–22 January for context.

The waveform families are **fine-scale, intermediate-scale, large-scale and
trend**. The first three are spatial band-pass components; the trend retains
the mean field. Their publication names are defined in
[waveform_bands.py](python/config/waveform_bands.py). Stable product identifiers
(`discontinuity`, `substructure`, `sheath`, `trend`) are retained in filenames
and saved interfaces; these identifiers map to the publication names and do
not identify physical sheath or ejecta intervals.

## Reproduce the analysis

Use Python 3.12 and the packages in
[python/requirements.txt](python/requirements.txt). On macOS, LightGBM also
requires OpenMP (`brew install libomp`). From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r python/requirements.txt
.venv/bin/python python/pipeline/run_publication_pipeline.py
```

The default input directory is `python/data/`. Outputs are written to
`python/output/`:

| Directory | Contents |
|---|---|
| `processed/` | Prepared observations, speed profiles, spatial coordinates and filtered waveforms |
| `results/` | Boundaries, metrics, comparisons, inference draws and GOES reconstructions |
| `figures/` | Eight main and five supplementary figures, all PDF |
| `figures/diag/` | Nine spacecraft overviews and three boundary diagnostics, all PDF |

For other input and output locations:

```bash
.venv/bin/python python/pipeline/run_publication_pipeline.py --data-dir /path/to/mission-inputs --output-dir /path/to/results
```

Keep the input and output directories separate, with neither nested inside the
other. Each full run recalculates derived products in dependency order and stops
if a required observation or calculation is missing or invalid. Supplied mission
files are read without modification.

Individual analysis and plotting scripts use `L1_GEO_DATA_DIR` and
`L1_GEO_OUTPUT_DIR` for these locations. They require the preceding products from
the same run. For example, to render Figure 2 after a completed calculation:

```bash
PYTHONPATH=python L1_GEO_DATA_DIR=/path/to/mission-inputs L1_GEO_OUTPUT_DIR=/path/to/results .venv/bin/python python/figures/fig_upstream_stack.py
```

The runtime pins SciPy 1.17.1. It avoids the older macOS wheel-loading
failure in SciPy 1.15.3. Coordinate conversion uses verified supplied
IGRF14 and UTC–TAI reference files and creates its SpacePy configuration
inside the output directory; no global SpacePy setup is required.

## Required observations

Mission observations must be supplied separately; they are not redistributed in
this repository. The input layout is below. `YYYYMMDD` and `*` denote dates and
provider/version suffixes. CDF and NetCDF files retain their original variables
and quality flags.

```text
mission-inputs/
  coordinate_reference/
    igrf14coeffs.txt
    tai-utc.dat
    manifest.json
  raw_event_20260119/
    solar_orbiter/
      solar_mag_YYYYMMDD.cdf
      solar_pas_YYYYMMDD.cdf
    l1/
      ac_h3_mfi_YYYYMMDD_v*.cdf
      wi_h0_mfi_YYYYMMDD_v*.cdf
      dscovr_h0s_mag_YYYYMMDD*.cdf    # preferred definitive product
      dscovr_h0_mag_YYYYMMDD_v*.cdf  # alternate supplied definitive flavor
      oe_f1m_dscovr_sYYYYMMDD*.nc.gz
      sscweb_step4_l1_lunar_20260119_1840_20260120_0800_gse.json
    earth_moon/
      thb_l2_fgm_YYYYMMDD_v01.cdf
      thb_l2_esa_YYYYMMDD_v01.cdf
      artemis_p2/
        thc_l2_fgm_YYYYMMDD_v01.cdf
        thc_l2_esa_YYYYMMDD_v01.cdf
      goes/
        dn_magn-l2-avg1m_g18_dYYYYMMDD_v*.nc
        dn_magn-l2-avg1m_g19_dYYYYMMDD_v*.nc
    scale_effects/goes_hires/
      dn_magn-l2-hires_g18_dYYYYMMDD_v*.nc
      dn_magn-l2-hires_g19_dYYYYMMDD_v*.nc
  ace/
    YYYYMMDD_ace_swepam_1m.txt
  wind/
    wind_h1s_swe_20260101000132_20260131235813_cdaweb.cdf
  swfo_mag_l3/
    sci_mag-l3_solar1_*.nc
  swfo_swips_l2/
    sci_swips-l2_*.nc
```

The Wind SWE input is the January monthly CDAWeb subset; daily
`wi_h1_swe_YYYYMMDD_v*.cdf` files are also accepted when the monthly file is absent. Solar-1 inputs are commissioning
products. The SSCWeb file provides L1, ARTEMIS and Moon positions; the Solar
Orbiter position is documented in the source configuration; the geometry
figure records its selected GOES reference positions.

The GOES orbit reader also accepts the supplied event-day names `g18_mag.nc`
and `g19_mag.nc`, checking each file's internal product identity and UTC day.

The corrected Solar-1 SWiPS delivery uses `sweep_mid_time`, `proton_n`,
`proton_v_gse` and `proton_t`. Density is in cm⁻³, velocity in km/s and
temperature in K; speed is the velocity-vector norm. The reader requires this
schema, the declared units and ordered times. The supplied `backfill`
indicator is not treated as a science-quality flag. Daily inputs must be
directly inside `swfo_swips_l2/`.

| Inputs | Required January 2026 coverage |
|---|---|
| Solar Orbiter MAG/PAS; ACE MAG/SWEPAM; Wind MAG; DSCOVR MAG/Faraday cup; Solar-1 MAG/SWiPS | 16–22 January |
| Wind SWE monthly subset | January, including 16–22 January |
| ARTEMIS-P1/P2 FGM | 17–22 January |
| ARTEMIS-P1/P2 ESA | 17–21 January |
| GOES-18/19 10-Hz MAG | 17–21 January |
| GOES-18/19 one-minute MAG orbit metadata (`orbit_llr_geo`) | 19–20 January |
| SSCWeb positions | 19 January 18:40 to 20 January 08:00 UTC |

January 17 and 22 magnetic observations provide filter padding. ARTEMIS
January 22 is magnetic-only padding. GOES January 17 and 18 provide same-UT
quiet references. January 16 upstream observations support the overviews and
speed preparation.

## Analysis methods

### Observation preparation

[Mission readers](python/ingest/missions.py) select one daily file version and
apply mission quality screening. Native timestamps and magnetic values are
retained for shock fits. The other representations serve distinct calculations:
common-cadence records contain arithmetic minute means, sample standard
deviations and counts; overview records contain medians; prepared one-second
magnetic grids use arithmetic means and the mission-specific hold rules described
in [the waveform methods](docs/filtering_method.md).

Bulk-speed preparation uses overview plasma medians and the local/proxy policies
in [build_speed_profiles.py](python/pipeline/build_speed_profiles.py). Proxy
shifts are calculated from measured shock times. Solar Orbiter vectors remain
RTN. Near-Earth magnetic components use native GSM products from ACE, Wind,
Solar-1, ARTEMIS and GOES. The original definitive CDAWeb DSCOVR `B1GSE`
vectors are converted to GSM at each native `Epoch1` timestamp with
[SpacePy CTrans](https://spacepy.github.io/coordinates.html), before cadence
preparation or filtering. DSCOVR's original `B1F1` scalar, `FLAG1` mask and
coverage are retained; NOAA's alternative MAG product is not substituted.
Other missions retain their existing independent scalar channels or the norm
of the original native GSE vector. Ephemerides and plasma velocity components
remain GSE. The scalar correlation method is unchanged.

The conversion requires `spacepy==0.7.0` and supplied physical reference files:
[NOAA/IAGA IGRF14 coefficients](https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt)
and [USNO UTC--TAI offsets](https://maia.usno.navy.mil/ser7/tai-utc.dat).
The `coordinate_reference/manifest.json` records schema version 1, the SpacePy
version, and a `references` list containing each `filename`, `runtime_filename`,
`source_url` and SHA-256. The runtime filenames are `igrfcoeffs.txt` and
`tai-utc.dat`, respectively. Missing inputs or checksum changes stop the run.
Verified copies and SpacePy configuration are confined to the current output
directory's `.cache/spacepy/`; no network access or global home configuration
is used. CTrans runs with `use_irbem=False` and `itol=0`, evaluates each native
UTC epoch without rotation interpolation, and rejects epochs outside the
supplied IGRF14 validity interval. Its default UT1--UTC offset is zero.

For example, after supplying the two linked files, the exact reference manifest
used for this run is below. These hashes identify physical model/time inputs,
not prior analysis outputs. Save this JSON as `coordinate_reference/manifest.json`
and verify the downloaded files match it before running the pipeline:

```json
{
  "schema_version": 1,
  "spacepy_version": "0.7.0",
  "references": [
    {
      "filename": "igrf14coeffs.txt",
      "runtime_filename": "igrfcoeffs.txt",
      "source_url": "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt",
      "sha256": "8f8d88403028fc4ee92c4f38d97b46e0a87e2cfc496045b43c9e26c1d6b0903c"
    },
    {
      "filename": "tai-utc.dat",
      "runtime_filename": "tai-utc.dat",
      "source_url": "https://maia.usno.navy.mil/ser7/tai-utc.dat",
      "sha256": "3524e1ae34d67e858873a89e59983bbc5bd100221da898e796c1b36036a310c3"
    }
  ]
}
```

An updated physical reference file requires its own documented hash in the
manifest; it must never be silently substituted for the supplied version.

A frame rotation preserves each unfiltered vector's norm. Because the GSE--GSM
rotation changes with time, transforming before filtering can alter filtered
vector metrics and directional boundary diagnostics. These quantities are
recalculated in the full run rather than assumed unchanged. Scalar quantities
remain suitable for comparisons including Solar Orbiter; RTN component signs
are not compared directly with GSM component signs.

### Physical boundaries

[Shock detection](python/analysis/shock_detection.py) scans the four-day magnetic
record with five-minute logistic fits to minute means, then 60-second fits to
native observations around candidates. The principal rising transition is
associated with the event within the arrival intervals in
[config/event_boundaries.py](python/config/event_boundaries.py), ranked by fitted
increase times R². Direct fits require R² ≥ 0.75. Within the L1 group and ARTEMIS
pair, a suitable neighboring detection and measured speed may guide a ballistic
search; the target's own native level crossings determine its time. ARTEMIS-P2
uses P1 guidance for this event. The method adapts
[Pelkum Donahue and Inceoglu (2026)](https://doi.org/10.3847/1538-4357/ae710c).

[Sheath–ejecta detection](python/analysis/sheath_ejecta_detection.py) uses observed
minute magnetic means. It identifies episodes where the centered ten-minute
standard deviation of |B| is below 2.5 nT for at least 80% of the following ten
minutes. The first L1 markers must contain a unique group of at least three
within ten minutes; an outlier uses its own episode nearest that group's frozen
median within ±10 minutes. Solar Orbiter and ARTEMIS retain local selections.
A ten-minute ejecta reference starts two minutes after the selected marker.
Angles to the normalized mean of unit magnetic vectors in that reference are
fitted with sliding ten-minute decreasing logistic transitions. Accepted fits
require R² ≥ 0.75, at least 80% observed samples, and three samples on each side
of the fitted 25–75% transition. Fits within five minutes are grouped; the
largest angular decrease times R² selects the boundary midpoint.

The sheath-entry timing half-width is the largest of half the fitted 25–75%
duration, one minute, and half the observed gap spanning the midpoint. It is a
descriptive width, not a confidence interval. The centered variability window
has ten minute samples; persistence, reference and fitting windows include both
endpoints. Missing bins do not qualify as quiet observations.

[Trailing magnetic recovery](python/analysis/ejecta_recovery.py) uses the |B|
95th percentile in the one-hour background ending two hours before the
shock-containing minute. After sheath entry, the first observed minute at or
below that level is accepted when at least eight of the ten bins beginning
there also qualify. The full ten-bin window must be present; missing bins fail.
The accepted window's beginning is the recovery time. Its cadence is one minute;
the confirmation duration is not a timing uncertainty, and no statistical
uncertainty is assigned.

Plasma first returns provide supporting measurements rather than gate magnetic
recovery. ACE plasma returns are excluded for the pre-shock tracking problem,
and DSCOVR post-shock plasma is excluded. ARTEMIS
plasma returns are descriptive because of mode/viewing effects and sparse speed.
The pre-shock background placement reflects inspection of this event. Boundaries,
fit evidence and diagnostic samples are saved in `results/`.

### Waveforms and GOES response

Figure 1 adds Earth/GOES insets and two Shue (1998) magnetopause outlines. The
[context stage](python/pipeline/build_fig1_magnetopause.py) recalculates paired
Solar-1 proton pressure and native GSM Bz states after the observed GOES response
product is generated. It uses the source-shock/response-onset clock difference
to associate the source minutes with the reference snapshot and immediate
post-onset context; it does not fit a new propagation time. The selected inputs,
pressure and surface parameters are saved in `results/fig1_magnetopause.csv`
and `.json`. The automatically generated `results/Fig1_caption.md` explains the
solid/dashed curves and retains method notes outside the figure artwork.

All four waveform families use the same prepared one-second magnetic input and
0.04-Earth-radius spatial grid. Reference scales are calculated from current-run
shock uncertainties, speeds at the four L1 shocks, sheath spectra and integrated
sheath/ejecta extents. Filtering precedes the calculation of minute means and
within-minute sample spreads. The [waveform methods](docs/filtering_method.md)
describe scale selection, gap handling, physical metrics and statistical inference.
Figure 4 uses the spatially weighted centered variance of the saved minute
scalar waveform divided by the sum of the three component variances. Each
channel is centered over its full selected waveform interval for this ratio.
Its other two rows give vector RMS amplitude and scalar-power width.

Figure 5 has three rows: peak lagged scalar Pearson correlation,
activity-weighted scalar magnitude-squared wavelet coherence, and GSM Bz Pearson
correlation at each pair's scalar-derived lag. Its three columns show the
band-pass families; the trend is omitted. Each panel displays all 15 near-Earth
pairs, while the same six L1--L1 and four L1--P1 links enter separate exponential
fits. P2 links provide context. Solar Orbiter remains in the underlying scalar
calculations and common wavelet support but is omitted from this figure.
The fitted labels are L_cor, L_coh and L_cor,z; fit qualifications are saved in
`results/Fig5_caption.md` and the numerical records.

The wavelet stage uses
each measured shock as the distance origin and applies one common four-L1
auto-power weight to all pairs, without optimizing another lag. Bz uses the
filtered one-minute GSM component; its bootstrap repeats the scalar lag search
rather than independently maximizing Bz correlation. The
[wavelet and Bz methods](docs/wavelet_coherence.md) distinguish bootstrap intervals
from conditional wavelet ranges and describe the limits of fitted distances.

The GOES response onset and transition are selected manually from the unfiltered
GOES-18/19 observations; the early-response interval spans fifteen minutes from
onset. Figure 6 includes seventeen minutes of pre-onset context and marks the
response onset separately from the later field transition. The observational
products cover that extended display, while the reconstruction models retain
the onset-to-end interval. A separate local-time stage converts the
native GOES one-minute orbit positions to solar magnetic coordinates using the
same verified SpacePy/IGRF14 reference inputs as the magnetic-vector conversion.
It recomputes the local-time record and the passages through dawn (06 MLT),
noon (12 MLT), dusk (18 MLT) and midnight (00 MLT), retaining passages within the
Figure 6 display. Spacecraft-coloured dash-dot lines and UTC labels show those
passages. January 17 and 18 observations provide
same-UT quiet references. The reconstruction targets are the raw GOES-18 and
GOES-19 field magnitudes, without quiet-reference subtraction. Each source uses
one common shift from its measured shock time to the GOES onset, rounded to
whole minutes. The same shift applies to every waveform family and component,
preserving their relative timing; there are no additional history inputs or
family-specific alignments.

The main P1 model uses twelve signed GSM component inputs: `bx_f`, `by_f` and
`bz_f` for each of the four waveform families. It does not use the scalar
waveforms or vector norms as additional predictors. Figures 7 and 8 show the
fits and TreeSHAP contributions grouped by family, by GSM axis and by their
individual family--component combinations. The two grouped views separately
reconstruct the same prediction when added to the model's expected output;
they are alternative partitions, not additive sets of contributions. Heatmaps
report each individual input's mean absolute contribution in nT.

Supplementary Figures 4 and 5 use four scalar `bt_f` waveform inputs from P1
and Solar-1, respectively. The scalar waveform is independently filtered
field magnitude, not the norm of a filtered vector. All models use the same
fixed LightGBM settings: 400 trees, learning rate 0.02, seven leaves, maximum
depth three, minimum leaf support fifteen samples and L1/L2 penalties 0.1/1.
Each source and GOES target is fitted separately over the observed event.

The JSON results include recomputed summaries for the three response intervals,
with signed ranges and mean absolute shares. Grouped shares sum the signed
TreeSHAP terms within each group before taking the mean absolute value.
These shares describe model reliance, not physical energy fractions or forecast
validation.

## Figures and results

| Figure | PDF in `figures/` |
|---|---|
| 1: observing geometry | `Fig1_geometry.pdf` |
| 2: event along the observing chain | `Fig2_upstream_bt_stack.pdf` |
| 3: aligned waveform families | `Fig3_isolated_waveforms.pdf` |
| 4: waveform extent, amplitude and variance ratio | `Fig4_physical_structure_evolution.pdf` |
| 5: scalar correlation/coherence and GSM Bz persistence | `Fig5_waveform_coherence_length_scales.pdf` |
| 6: observed GOES magnetic response | `Fig6_goes_magnetic_response.pdf` |
| 7: P1 component-input reconstruction and grouped contributions | `Fig7_p1_goes_reconstruction.pdf` |
| 8: P1 component contributions within each waveform family | `Fig8_p1_goes_component_contributions.pdf` |
| Supplementary 1: band-energy intervals used in Figure 4 | `SupFig1_extraction_intervals.pdf` |
| Supplementary 2: lag and coherence | `SupFig2_lag_coherence.pdf` |
| Supplementary 3: amplitude ratios and normalized differences | `SupFig3_information_retention.pdf` |
| Supplementary 4: P1 scalar-input GOES reconstruction | `SupFig4_p1_scalar_reconstruction.pdf` |
| Supplementary 5: Solar-1 scalar-input GOES reconstruction | `SupFig5_solar1_reconstruction.pdf` |

`figures/diag/` contains the nine spacecraft overviews and shock, sheath-entry
and recovery diagnostics. Principal numerical summaries in `results/` are
`boundaries.json`, `waveform_intervals.json`, `chain_waveform_metrics.json`,
`waveform_inference.json`, `wavelet_coherence.json`, `bz_persistence.json`,
`paired_goes_response.json`,
`p1_goes_reconstruction.json`, `p1_scalar_goes_reconstruction.json` and
`solar1_goes_reconstruction.json`, with
corresponding tables and resampling draws. Bz also writes
`bz_persistence.csv`, `bz_persistence_fits.csv` and `bz_persistence_draws.npz`;
these are recomputed from the current filtered GSM components and scalar lags.
Each reconstruction's
`*_event_samples.csv` combines observations, predictions and TreeSHAP
contributions. Quiet references remain observational context rather than
reconstruction targets.

## Code structure

`python/common/` provides shared file and UTC helpers, `ingest/` reads mission
products, `analysis/` contains numerical methods,
`config/` holds method parameters, `pipeline/` runs the calculations, and
`figures/` renders their products. Generated files belong in `python/output/`.
The complete calculation order is specified by
[run_publication_pipeline.py](python/pipeline/run_publication_pipeline.py).

The public workflow contains runtime checks for required variables, units,
ordered timestamps, usable data coverage and valid numerical operations.
Every full run starts from supplied observations and regenerates all derived
products. No previous output files are required. All thirteen paper figures
and twelve inspection PDFs are generated by the same command. The recovery
inspection shows magnetic field, field direction, speed and temperature;
its end time is selected by the stated magnetic recovery rule.

Study PI: **Fadil İnceoğlu**
