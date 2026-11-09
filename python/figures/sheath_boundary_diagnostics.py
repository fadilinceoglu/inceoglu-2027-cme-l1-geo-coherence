"""Show the registered variability selection, reference and angular fit."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from analysis.event_boundaries import sheath_measurement
from analysis.shock_detection import logistic
from config import study
from config.paths import RESULTS_DIR, DIAGNOSTIC_DIR
from config.event_boundaries import UPSTREAM_SPACECRAFT
from figures.style import apply_style


def plot_page(pdf, craft, record, report):
    fig, axes = plt.subplots(4, 1, figsize=(8.3, 10.7))
    fig.subplots_adjust(left=.11, right=.96, top=.90, bottom=.17, hspace=.49)
    fig.suptitle(f"{study.SATELLITES[craft]['name']}: sheath-ejecta boundary", fontsize=14, y=.971)
    fig.text(.11, .935, "Observed one-minute magnetic records; data-selected reference and logistic angular fit", fontsize=9)
    lo = report['shock_utc'] - pd.Timedelta('5min')
    hi = report['reference_stop_utc'] + pd.Timedelta('3min')
    data = record.loc[lo:hi]
    axes[0].plot(data.index, data.bt, color='#343434', lw=.8)
    axes[0].set_ylabel(r'$|B|$ (nT)')
    axes[0].set_title('(a) Magnetic magnitude', loc='left', fontsize=10)
    axes[1].plot(data.index, data.moving_std_nt, color='#7B3294', lw=.9)
    axes[1].axhline(report['settings']['variability_threshold_nt'], color='#7B3294', ls='--', lw=.8)
    settings = report['settings']
    axes[1].set_ylabel(rf"$s_{{|B|,{settings['variability_window_min']}}}$ (nT)")
    axes[1].set_title(
        f"(b) Sustained low variability: <{settings['variability_threshold_nt']:g} nT, "
        f"{settings['persistence_fraction']:.0%} of the following {settings['persistence_duration_min']} min",
        loc='left', fontsize=10)

    selection = report['selection']
    if selection['status'] == 'fleet_reselected':
        first = pd.Timestamp(selection['first_marker_utc'])
        axes[1].axvline(first, color='#B35806', ls='-.', lw=.9)
        axes[1].text(.02, .94, f"L1 check: first {first:%H:%M}; selected {report['variability_marker_utc']:%H:%M} UT",
                     transform=axes[1].transAxes, va='top', fontsize=8,
                     bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    axes[2].plot(data.index, data.theta_ejecta_deg, color='#2166AC', lw=.8)
    axes[2].set_ylabel(r'$\theta_{\rm ej}$ (deg)')
    axes[2].set_title('(c) Angle to the ejecta reference throughout the search', loc='left', fontsize=10)
    axes[2].set_ylim(-5, 185)
    mid = report['boundary_utc']
    a, b = report['fit_window_start_utc'], report['fit_window_stop_utc']
    for ax in axes[:3]:
        ax.axvspan(report['reference_start_utc'], report['reference_stop_utc'], color='#2166AC', alpha=.12, lw=0)
        ax.axvline(report['variability_marker_utc'], color='#7B3294', ls=':', lw=1)
        ax.axvline(mid, color='#D95F02', lw=1.1)
        ax.axvspan(a, b, color='#D95F02', alpha=.07, lw=0)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel(f'{mid:%d %B %Y} (UTC)', fontsize=8)
    ax = axes[3]
    ax.set_title('(d) Observed minute angles and the selected logistic fit', loc='left', fontsize=10)
    near = record.loc[a - pd.Timedelta('3min'):b + pd.Timedelta('3min')]
    ax.plot(near.index, near.theta_ejecta_deg, color='#AAAAAA', lw=.8)
    obs = record.loc[a:b].theta_ejecta_deg
    ax.scatter(obs.index, obs, s=19, color='#2166AC', label='Observed angles', zorder=3)
    grid = pd.date_range(a, b, periods=601)
    relative = (grid - mid).total_seconds().to_numpy() / 60
    prediction = logistic(relative, report['before_deg'], report['after_deg'], 0., np.log(report['tau_minutes']))
    ax.plot(grid, prediction, color='#D95F02', lw=1.5, label='Logistic fit', zorder=4)
    ax.axvline(mid, color='#D95F02', lw=1.2, label='Fitted midpoint')
    ax.axvspan(report['transition_25_utc'], report['transition_75_utc'], color='#D95F02', alpha=.14, lw=0)
    if report['gap_limited']:
        ax.axvspan(report['last_sample_before_utc'], report['first_sample_after_utc'],
                   color='#888888', alpha=.18, lw=0, label='Gap spanning midpoint')
    ax.set_ylim(-5, 185)
    ax.set_ylabel(r'$\theta_{\rm ej}$ (deg)')
    ax.legend(loc='upper right', fontsize=7.4, frameon=True, facecolor='white', framealpha=.93)
    ax.set_xlim(near.index.min(), near.index.max())
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel(f'{mid:%d %B %Y} (UTC)', fontsize=8)
    fig.text(.11, .113, f"Midpoint: {mid.round('s'):%H:%M:%S} UT +/- {report['timing_halfwidth_seconds']:.0f} s; "
             f"R-squared = {report['r2']:.4f}; fitted angular decrease = {report['drop_deg']:.1f} deg", fontsize=9)
    fig.text(.11, .091, f"Reference: {report['reference_start_utc']:%H:%M}-{report['reference_stop_utc']:%H:%M} UT; "
             f"90% of directions within {report['reference_angle_p90_deg']:.1f} deg of its mean", fontsize=8.5)
    fig.text(.11, .069, f"Fitted 25-75% duration: {report['transition_25_75_seconds']:.1f} s. "
             + ('Width below one-minute sampling. ' if report['sampling_limited'] else '')
             + ('Midpoint spans a data gap.' if report['gap_limited'] else ''), fontsize=8)
    fig.text(.11, .039, 'Blue shading: reference. Purple dotted: variability marker. Orange: angular fit.\n'
             'Timing half-width includes cadence and gap floors; it is not a confidence interval.',
             fontsize=7.7, color='#555555', linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)


def build(output=DIAGNOSTIC_DIR / 'sheath_ejecta_boundaries.pdf'):
    apply_style()
    frame = pd.read_csv(RESULTS_DIR / 'sheath_boundary_diagnostics.csv', parse_dates=['time'])
    frame['time'] = pd.to_datetime(frame.time, utc=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for craft in UPSTREAM_SPACECRAFT:
            record = frame.loc[frame.spacecraft.eq(craft)].set_index('time')
            plot_page(pdf, craft, record, sheath_measurement(craft))
    return output


if __name__ == '__main__':
    print(build())
