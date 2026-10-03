"""
pipeline.py - TESS transit search pipeline (Phase 2).

Flow for one star:
    fetch -> clean -> detrend -> BLS search -> refine
          -> mask transits, re-detrend, refine again -> measure

Everything learned in notebook 01 (finding planets by hand) is automated here:
  * instrument glitches after data gaps are trimmed automatically
  * only UPWARD outliers are clipped (transits are downward, so a
    symmetric sigma clip would delete deep transits like WASP-18 b's)
  * the period grid is sized to the data's time baseline
  * coarse-to-fine period search
  * two-pass detrending with the transits masked
"""
from __future__ import annotations

import re

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

from .features import compute_features

DEFAULT_DURATIONS = (0.04, 0.06, 0.08, 0.10, 0.15, 0.20)  # trial transit durations, days


def _val(x):
    """Strip astropy units/Time wrappers, returning a plain number."""
    return getattr(x, "value", x)


# Fetch

def _closest_sectors(search, max_sectors):
    """Indices of the `max_sectors` search results whose sector numbers are
    closest together. Sectors far apart in time make the period grid (and
    runtime) explode without adding more transits than nearby sectors would."""
    sectors = np.array([int(re.search(r"(\d+)", str(m)).group(1)) for m in search.mission])
    order = np.argsort(sectors)
    k = min(max_sectors, len(order))
    spans = [sectors[order[i + k - 1]] - sectors[order[i]] for i in range(len(order) - k + 1)]
    best = int(np.argmin(spans))
    idx = sorted(order[best:best + k].tolist())
    return idx, sorted(sectors[idx].tolist())


def fetch_lightcurve(target, max_sectors=1, author="SPOC", exptime=120):
    """Download and stitch up to `max_sectors` TESS sectors for a target,
    choosing the sectors closest together in time.

    Returns (lightcurve, number_of_sectors_used, list_of_sector_numbers).
    """
    search = lk.search_lightcurve(target, mission="TESS", author=author, exptime=exptime)
    if len(search) == 0:
        raise ValueError(f"No TESS {author} {exptime}s light curves found for {target!r}")

    idx, sectors = _closest_sectors(search, max_sectors)
    chosen = lk.SearchResult(search.table[idx])
    collection = chosen.download_all(quality_bitmask="hard")
    if collection is None or len(collection) == 0:
        raise RuntimeError(f"Download failed for {target!r}")

    lc = collection.stitch().remove_nans()
    return lc, len(collection), sectors


# Clean

def clean(lc, gap_days=0.5, trim_days=1.0, min_segment_days=1.0, sigma_upper=5.0):
    """Automatic version of the manual glitch removal from Phase 1.

    - Splits the light curve into segments wherever there is a gap
      longer than `gap_days` (sector boundaries, downlinks, instrument
      interruptions).
    - Trims the first `trim_days` of every segment, where detector
      settling ramps and spikes usually live.
    - Drops any segment left shorter than `min_segment_days`
      (e.g. the short glitchy island in Pi Mensae's Sector 4).
    - Clips only upward outliers, so transits are never removed.
    """
    t = lc.time.value
    seg_id = np.concatenate([[0], np.cumsum(np.diff(t) > gap_days)])
    keep = np.ones(len(t), dtype=bool)

    for s in np.unique(seg_id):
        in_seg = seg_id == s
        t_seg = t[in_seg]
        start = t_seg[0]
        keep &= ~(in_seg & (t < start + trim_days))
        if (t_seg[-1] - start - trim_days) < min_segment_days:
            keep &= ~in_seg

    lc = lc[keep]
    return lc.remove_outliers(sigma_lower=np.inf, sigma_upper=sigma_upper)


# Detrend

def detrend(lc, window_length=901, mask=None):
    """Remove slow stellar/instrumental trends. `mask` (True = in transit)
    excludes those points from the trend fit."""
    return lc.flatten(window_length=window_length, mask=mask)

# Period search

def period_grid(time, pmin, pmax, dur_min, oversample=3):
    """Log-uniform period grid, sized to the time baseline.

    Over a baseline T, a period error dP shifts the last transit by
    (T/P)*dP. Keeping that shift below the shortest trial duration
    requires dP/P < duration/T, i.e. a fixed step in log(period).
    Longer baselines -> finer grid -> more compute.
    """
    baseline = time.max() - time.min()
    pmax = min(pmax, baseline / 2)  # require at least two transits
    dlnp = dur_min / (baseline * oversample)
    n = int(np.ceil(np.log(pmax / pmin) / dlnp))
    return np.exp(np.linspace(np.log(pmin), np.log(pmax), n))


# lightkurve estimates the size of ITS OWN default grid before every search
# and refuses to run if that estimate is huge, even when we pass our own
# period array. Its estimate grows with baseline**2, so multi-year data trips
# it. A large frequency_factor shrinks that estimate. It has no other effect
# when an explicit period grid is supplied.

_SKIP_LK_GRID_CHECK = {"frequency_factor": 1e4}


def bls_search(flat, pmin=0.5, pmax=15.0, durations=DEFAULT_DURATIONS, oversample=3):
    """Coarse BLS search over the full period range."""
    periods = period_grid(flat.time.value, pmin, pmax, min(durations), oversample)
    return flat.to_periodogram(method="bls", period=periods, duration=list(durations),
                               **_SKIP_LK_GRID_CHECK)


def refine_period(flat, period, durations=DEFAULT_DURATIONS, width_frac=0.002, n=3000):
    """Fine BLS search in a narrow window around a candidate period."""
    fine_periods = np.linspace(period * (1 - width_frac), period * (1 + width_frac), n)
    fine_durations = np.linspace(min(durations), max(durations), 15)
    return flat.to_periodogram(method="bls", period=fine_periods, duration=fine_durations,
                               **_SKIP_LK_GRID_CHECK)


def _best(bls):
    """(period, t0, duration) at the periodogram's highest peak, as floats."""
    return (float(_val(bls.period_at_max_power)),
            float(_val(bls.transit_time_at_max_power)),
            float(_val(bls.duration_at_max_power)))


# Measure

def measure_transit(flat, period, t0, duration):
    """Depth, noise and signal-to-noise from the folded light curve."""
    folded = flat.fold(period=period, epoch_time=t0)
    phase = np.asarray(folded.time.value, dtype=float)
    flux = np.asarray(folded.flux.value, dtype=float)

    core = np.abs(phase) < duration / 4   # central half of the transit
    out = np.abs(phase) > duration        # clearly out of transit

    base = np.median(flux[out])
    depth = base - np.median(flux[core])
    noise = 1.4826 * np.median(np.abs(flux[out] - base))  # robust per-point scatter
    n_core = int(core.sum())
    snr = depth / (noise / np.sqrt(n_core)) if (n_core > 0 and noise > 0) else np.nan

    # How many individual transits actually have data (>= 5 points in transit).
    # Detections built on only 1-2 transits are much less reliable.
    t = np.asarray(flat.time.value, dtype=float)
    epoch = np.round((t - t0) / period)
    in_tr = np.abs(t - (t0 + epoch * period)) < duration / 2
    _, counts = np.unique(epoch[in_tr], return_counts=True)
    n_transits = int(np.sum(counts >= 5))

    return {
        "depth_ppm": depth * 1e6,
        "noise_ppm": noise * 1e6,
        "n_in_transit": n_core,
        "n_transits": n_transits,
        "snr": snr,
    }

# Full pipeline for one star

def process_star(target, max_sectors=1, pmin=0.5, pmax=15.0,
                 durations=DEFAULT_DURATIONS, return_data=False):
    """Run the whole pipeline on one star and return a summary dict.

    With return_data=True, also returns the intermediate light curves and
    periodogram for plotting: (summary, data).
    """
    lc_raw, n_sectors, sectors = fetch_lightcurve(target, max_sectors)
    lc = clean(lc_raw)

    # Pass 1: detrend blind, search, refine
    flat = detrend(lc)
    bls = bls_search(flat, pmin, pmax, durations)
    period, t0, duration = _best(refine_period(flat, float(_val(bls.period_at_max_power)), durations))

    # Pass 2: mask the transits (2x BLS duration, since BLS underestimates it),
    # re-detrend so the trend can't bend into them, refine again
    mask = lc.create_transit_mask(period=period, transit_time=t0, duration=2 * duration)
    flat = detrend(lc, mask=mask)
    period, t0, duration = _best(refine_period(flat, period, durations))

    # Reliability flags
    baseline = float(lc.time.value.max() - lc.time.value.min())
    pmax_searched = min(pmax, baseline / 2)   # same cap as period_grid()
    at_edge = bool(period < pmin * 1.01 or period > pmax_searched * 0.99)

    summary = {
        "target": target,
        "tic_id": lc_raw.meta.get("TICID"),
        "n_sectors": n_sectors,
        "sectors": " ".join(str(s) for s in sectors),
        "n_points": len(lc),
        "baseline_days": baseline,
        "pmax_searched": pmax_searched,
        "period_days": period,
        "t0_btjd": t0,
        "duration_days": duration,
        "at_search_edge": at_edge,
        "bls_power": float(_val(bls.max_power)),
        **measure_transit(flat, period, t0, duration),
        **compute_features(lc, flat, period, t0, duration),
    }

    if return_data:
        return summary, {"lc_raw": lc_raw, "lc": lc, "flat": flat, "bls": bls}
    return summary

# Diagnostic plot

def plot_summary(summary, data, save_path=None, show=True):
    """Three-panel diagnostic: light curve, periodogram, folded transit.

    save_path: if given, the figure is also saved there as an image.
    show: set False in batch runs so figures are saved but not displayed.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    data["lc_raw"].plot(ax=axes[0], color="lightgrey", label="raw (stitched)")
    data["lc"].plot(ax=axes[0], label="cleaned")
    axes[0].set_title(f"{summary['target']}   (TIC {summary['tic_id']}, "
                      f"{summary['n_sectors']} sector(s))")

    data["bls"].plot(ax=axes[1])
    axes[1].axvline(summary["period_days"], color="red", ls="--", alpha=0.5)
    axes[1].set_title("BLS periodogram (pass 1)")

    folded = data["flat"].fold(period=summary["period_days"], epoch_time=summary["t0_btjd"])
    folded.scatter(ax=axes[2], alpha=0.2)
    folded.bin(time_bin_size=summary["duration_days"] / 10).plot(ax=axes[2], color="red", lw=2)
    w = 3 * summary["duration_days"]
    d = summary["depth_ppm"] * 1e-6
    nz = summary["noise_ppm"] * 1e-6
    axes[2].set_xlim(-w, w)
    axes[2].set_ylim(1 - 2 * d - 3 * nz, 1 + d + 3 * nz)
    axes[2].set_title(f"P = {summary['period_days']:.5f} d   "
                      f"depth = {summary['depth_ppm']:.0f} ppm   "
                      f"SNR = {summary['snr']:.1f}")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)  # free memory when processing many stars
