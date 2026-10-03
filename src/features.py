"""
features.py - vetting features for the planet vs false-positive classifier.

Every feature here is measured by the pipeline from the light curve, or
derived from independent stellar properties. Catalog periods/depths are
never used as features: that would leak the answer into the model.

Light-curve features (compute_features):
    duty_cycle
        Transit duration / period. Very large values = not a transit.
    depth_odd_ppm, depth_even_ppm, odd_even_sigma
        Binaries found at half their period alternate deep/shallow eclipses.
    secondary_depth_ppm, secondary_sigma, secondary_ratio
        A dip at phase 0.5 (the companion passing BEHIND) points to a binary,
        unless it is tiny relative to the transit (hot Jupiters glow too).
    shape_ratio
        Depth in the outer half of the transit / depth in the inner half.
        U-shaped planet transits ~0.8-1.0 whereas V-shaped binaries ~0.3-0.5.
    red_noise_beta, snr_red
        How correlated the noise is on the transit timescale, and an SNR
        that accounts for it (variable stars stop getting inflated scores).
    depth_chi2, max_transit_frac
        Are individual transits consistent with each other? Is the whole
        signal carried by one event?
    variability_ppm
        Amplitude of the star's own variability before detrending.
    centroid_offset_px, centroid_sigma, centroid_source_offset_px
        Does the light's centre move during transit? Scaled by depth, how far
        from the target is the source of the dip? (background binaries)

Derived from stellar properties (add_derived_features):
    planet_radius_re, expected_duration_days, duration_ratio
"""
from __future__ import annotations

import numpy as np

FEATURE_COLUMNS = [
    "duty_cycle",
    "depth_odd_ppm", "depth_even_ppm", "odd_even_sigma",
    "secondary_depth_ppm", "secondary_sigma", "secondary_ratio",
    "shape_ratio",
    "red_noise_beta", "snr_red",
    "depth_chi2", "max_transit_frac",
    "variability_ppm",
    "centroid_offset_px", "centroid_sigma", "centroid_source_offset_px",
]

DERIVED_COLUMNS = ["planet_radius_re", "expected_duration_days", "duration_ratio"]


def _robust_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _segments(t, gap_days=0.5):
    """Integer label for each continuous stretch of data."""
    return np.concatenate([[0], np.cumsum(np.diff(t) > gap_days)])


def _phase(t, t0, period):
    """Time from nearest mid-transit (days) and the transit number."""
    phase = (t - t0 + 0.5 * period) % period - 0.5 * period
    epoch = np.floor((t - t0 + 0.5 * period) / period).astype(int)
    return phase, epoch


def _depth(f, mask, base, noise):
    """Depth below baseline and its uncertainty (error of a median)."""
    n = int(mask.sum())
    if n < 3 or not np.isfinite(noise):
        return np.nan, np.nan
    return base - np.median(f[mask]), 1.2533 * noise / np.sqrt(n)


def compute_features(lc, flat, period, t0, duration):
    """Vetting features for one detected signal.

    lc   : cleaned light curve BEFORE detrending (for variability, centroids)
    flat : detrended light curve used for the search
    """
    feats = {k: np.nan for k in FEATURE_COLUMNS}

    t = np.asarray(flat.time.value, dtype=float)
    f = np.asarray(flat.flux.value, dtype=float)
    P, T = period, duration
    phase, epoch = _phase(t, t0, P)
    dist_sec = 0.5 * P - np.abs(phase) # distance from phase 0.5

    core = np.abs(phase) < T / 4  # inner half of the transit
    ring = (np.abs(phase) >= T / 4) & (np.abs(phase) < T / 2)
    sec = dist_sec < T / 4 # centre of possible secondary
    out = (np.abs(phase) > T) & (dist_sec > T) # clean baseline
    if out.sum() < 10:
        # Very long "transits" relative to the period leave no clean
        # baseline; fall back to a narrower exclusion zone.
        out = (np.abs(phase) > T / 2) & (dist_sec > T / 2)

    # Fraction of the orbit spent in transit. Planets: ~0.01-0.1.
    # Much larger values mean continuous variation, not a transit.
    feats["duty_cycle"] = T / P

    if out.sum() < 10 or core.sum() < 3:
        return feats

    base = np.median(f[out])
    noise = _robust_std(f[out])
    d_core, _ = _depth(f, core, base, noise)

    # Odd vs even transits
    d_odd, e_odd = _depth(f, core & (epoch % 2 == 1), base, noise)
    d_even, e_even = _depth(f, core & (epoch % 2 == 0), base, noise)
    feats["depth_odd_ppm"] = d_odd * 1e6
    feats["depth_even_ppm"] = d_even * 1e6
    if np.isfinite(d_odd) and np.isfinite(d_even):
        feats["odd_even_sigma"] = abs(d_odd - d_even) / np.hypot(e_odd, e_even)

    # Secondary eclipse at phase 0.5 (assumes a circular orbit)
    d_sec, e_sec = _depth(f, sec, base, noise)
    feats["secondary_depth_ppm"] = d_sec * 1e6
    if np.isfinite(d_sec):
        feats["secondary_sigma"] = d_sec / e_sec
        if d_core > 0:
            # Hot Jupiters can show a tiny secondary (a few % of the transit);
            # binaries of similar stars can reach tens of percent.
            feats["secondary_ratio"] = d_sec / d_core

    # Shape: U vs V 
    d_ring, _ = _depth(f, ring, base, noise)
    if np.isfinite(d_ring) and d_core > 0:
        feats["shape_ratio"] = d_ring / d_core

    # Per-transit consistency 
    depths, errs = [], []
    for e in np.unique(epoch[core]):
        d, err = _depth(f, core & (epoch == e), base, noise)
        if np.isfinite(d):
            depths.append(d)
            errs.append(err)
    depths, errs = np.array(depths), np.array(errs)
    n_tr = len(depths)
    if n_tr >= 2:
        w = 1 / errs**2
        mean = np.sum(w * depths) / np.sum(w)
        feats["depth_chi2"] = np.sum(((depths - mean) / errs) ** 2) / (n_tr - 1)
    if n_tr >= 1:
        s2 = np.clip(depths / errs, 0, None) ** 2
        if s2.sum() > 0:
            feats["max_transit_frac"] = s2.max() / s2.sum()

    # Correlated ("red") noise on the transit timescale 
    # Average the out-of-transit flux in bins as long as the transit core.
    # With purely random noise the binned scatter would be noise/sqrt(N);
    # beta = actual / expected (1 = white noise, >1 = correlated noise).
    cadence = np.median(np.diff(t))
    bin_w = T / 2
    t_out, f_out = t[out], f[out]
    b = np.floor(t_out / bin_w).astype(int)
    b -= b.min()
    sums = np.bincount(b, weights=f_out)
    cnts = np.bincount(b)
    sel = cnts >= 0.5 * (bin_w / cadence)
    if sel.sum() >= 10 and n_tr >= 1:
        means = sums[sel] / cnts[sel]
        sigma_bin = _robust_std(means)
        expected = noise / np.sqrt(np.median(cnts[sel]))
        feats["red_noise_beta"] = sigma_bin / expected
        feats["snr_red"] = d_core / (sigma_bin / np.sqrt(n_tr))

    # Stellar variability (before detrending, transits excluded)
    t_raw = np.asarray(lc.time.value, dtype=float)
    raw = np.asarray(lc.flux.value, dtype=float)
    ph_raw, _ = _phase(t_raw, t0, P)
    raw = raw[(np.abs(ph_raw) > T) & np.isfinite(raw)]
    if len(raw) > 10:
        feats["variability_ppm"] = (np.percentile(raw, 95) - np.percentile(raw, 5)) * 1e6

    # Centroid shift during transit 
    if "centroid_col" in lc.columns and "centroid_row" in lc.columns:
        tc = np.asarray(lc.time.value, dtype=float)
        ph_c, _ = _phase(tc, t0, P)
        dist_c = 0.5 * P - np.abs(ph_c)
        core_c = np.abs(ph_c) < T / 4
        out_c = (np.abs(ph_c) > T) & (dist_c > T)
        if out_c.sum() < 10:
            out_c = (np.abs(ph_c) > T / 2) & (dist_c > T / 2)
        seg = _segments(tc)
        shifts, sigmas = [], []
        for name in ("centroid_col", "centroid_row"):
            x = np.asarray(lc[name].value, dtype=float)
            # Each sector/orbit puts the star on different pixels: remove
            # each segment's median position before comparing.
            for s in np.unique(seg):
                m = seg == s
                x[m] = x[m] - np.nanmedian(x[m])
            ok_core = core_c & np.isfinite(x)
            ok_out = out_c & np.isfinite(x)
            if ok_core.sum() >= 3 and ok_out.sum() >= 10:
                shift = np.mean(x[ok_core]) - np.mean(x[ok_out])
                err = _robust_std(x[ok_out]) / np.sqrt(ok_core.sum())
                shifts.append(shift)
                sigmas.append(shift / err if err > 0 else np.nan)
        if len(shifts) == 2:
            offset = float(np.hypot(*shifts))
            feats["centroid_offset_px"] = offset
            feats["centroid_sigma"] = float(np.hypot(*sigmas))
            # Any deep transit nudges the centroid a little if other stars share
            # the aperture. Dividing by the depth estimates how far the source
            # of the dip is from the target (~0 = the target itself).
            if d_core > 0:
                feats["centroid_source_offset_px"] = offset / d_core

    return feats

def add_derived_features(df):
    """Features that combine pipeline measurements with stellar properties
    (st_rad in solar radii, st_logg in cgs) from the TOI table."""
    df = df.copy()
    depth = df["depth_ppm"].clip(lower=0) * 1e-6

    # depth ~ (Rp/R*)^2  ->  Rp = sqrt(depth) * R*   (1 R_sun = 109.1 R_earth)
    df["planet_radius_re"] = np.sqrt(depth) * df["st_rad"] * 109.1

    # Expected duration of a central transit on a circular orbit:
    # T ~ 13 h * (P / 1 yr)^(1/3) * (rho_star / rho_sun)^(-1/3)
    # with density from surface gravity and radius: rho ~ g / R
    rho = 10 ** (df["st_logg"] - 4.438) / df["st_rad"]
    df["expected_duration_days"] = (13 / 24) * (df["period_days"] / 365.25) ** (1 / 3) * rho ** (-1 / 3)
    df["duration_ratio"] = df["duration_days"] / df["expected_duration_days"]
    return df
