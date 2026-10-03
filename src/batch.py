"""
batch.py - run the pipeline over many TESS targets and check it against
the TESS Objects of Interest (TOI) catalog.

    load_toi_catalog()  -> the full TOI table from the NASA Exoplanet Archive
    select_sample()     -> a labelled sample of planets and false positives
    run_batch()         -> process every star, saving results as it goes
    evaluate()          -> compare recovered periods with the catalog
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from .pipeline import process_star, plot_summary
from .features import FEATURE_COLUMNS

# Project folders, worked out from this file's location (src/batch.py),
# so paths are correct no matter where the notebook is run from

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TOI_COLUMNS = ["toi", "tid", "tfopwg_disp", "pl_orbper", "pl_trandurh",
               "pl_trandep", "st_tmag", "st_rad", "st_logg", "st_teff"]
PLANET_DISPS = ["CP", "KP"]   # confirmed planet, known planet
FP_DISPS = ["FP"]             # false positive

# Fixed column order for the results file, so rows that errored (and have
# fewer fields) still line up with successful ones

RESULT_COLUMNS = [
    "target", "toi", "label", "catalog_period", "catalog_depth_ppm",
    "status", "error", "runtime_s",
    "tic_id", "n_sectors", "sectors", "n_points", "baseline_days", "pmax_searched",
    "period_days", "t0_btjd", "duration_days", "at_search_edge",
    "bls_power", "depth_ppm", "noise_ppm", "n_in_transit", "n_transits", "snr",
    "st_tmag", "st_rad", "st_logg", "st_teff",
] + FEATURE_COLUMNS

# Catalog

def load_toi_catalog(refresh=False):
    """Download the TOI table (cached to data/toi_catalog.csv)."""
    cache = DATA_DIR / "toi_catalog.csv"
    if cache.exists() and not refresh:
        toi = pd.read_csv(cache)
        if all(c in toi.columns for c in TOI_COLUMNS):
            return toi
        # cached copy predates newly added columns -> download again

    query = f"select {', '.join(TOI_COLUMNS)} from toi"
    url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
           + quote(query) + "&format=csv")
    toi = pd.read_csv(url)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    toi.to_csv(cache, index=False)
    return toi

# Sample

def select_sample(toi, n_planets=50, n_fps=50, pmin=0.5, pmax=15.0,
                  max_tmag=11.0, seed=42):
    """Pick a labelled, reproducible sample of planets and false positives.

    Filters:
      - single-TOI stars only (multi-planet systems are handled later)
      - catalog period within the pipeline's search range
      - TESS magnitude <= max_tmag (brighter stars, cleaner data)
    """
    df = toi.copy()

    counts = df["tid"].value_counts()
    df = df[df["tid"].map(counts) == 1]
    df = df[df["pl_orbper"].between(pmin, pmax) & (df["st_tmag"] <= max_tmag)]

    planets = df[df["tfopwg_disp"].isin(PLANET_DISPS)]
    fps = df[df["tfopwg_disp"].isin(FP_DISPS)]

    sample = pd.concat([
        planets.sample(min(n_planets, len(planets)), random_state=seed),
        fps.sample(min(n_fps, len(fps)), random_state=seed),
    ])
    sample["label"] = np.where(sample["tfopwg_disp"].isin(PLANET_DISPS),
                               "planet", "false_positive")
    sample["target"] = "TIC " + sample["tid"].astype(int).astype(str)
    return sample.reset_index(drop=True)

# Batch Run

def run_batch(sample, name="batch1", max_sectors=1, save_plots=True):
    """Run process_star on every target in `sample`.

    - One failing star never stops the run: its error is recorded instead.
    - Each result is appended to data/<name>/results.csv immediately, so if
      the run is interrupted (or you stop it), running again resumes where
      it left off.
    - Diagnostic plots are saved to data/<name>/plots/.
    - When finished, everything is also written to results.parquet.
    """
    out_dir = DATA_DIR / name
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "results.csv"

    done = set(pd.read_csv(progress)["target"]) if progress.exists() else set()
    todo = sample[~sample["target"].isin(done)]
    print(f"{len(done)} already done, {len(todo)} to go\n")

    for i, row in enumerate(todo.itertuples(index=False), 1):
        start = time.time()
        record = {
            "target": row.target,
            "toi": row.toi,
            "label": row.label,
            "catalog_period": row.pl_orbper,
            "catalog_depth_ppm": row.pl_trandep,
            "st_tmag": row.st_tmag,
            "st_rad": row.st_rad,
            "st_logg": row.st_logg,
            "st_teff": row.st_teff,
        }
        try:
            summary, data = process_star(row.target, max_sectors=max_sectors,
                                         return_data=True)
            record.update(summary)
            record["status"] = "ok"
            if save_plots:
                png = plot_dir / f"{row.target.replace(' ', '_')}.png"
                plot_summary(summary, data, save_path=png, show=False)
        except Exception as e:
            record["status"] = "error"
            record["error"] = f"{type(e).__name__}: {e}"

        record["runtime_s"] = round(time.time() - start, 1)
        (pd.DataFrame([record])
           .reindex(columns=RESULT_COLUMNS)
           .to_csv(progress, mode="a", header=not progress.exists(), index=False))
        print(f"[{i:>3}/{len(todo)}] {row.target:<16} {record['status']:<6} "
              f"{record['runtime_s']:>6}s")

    results = pd.read_csv(progress)
    results.to_parquet(out_dir / "results.parquet", index=False)
    print(f"\nSaved {len(results)} results to {out_dir}")
    return results

# Evaluation

ALIAS_RATIOS = [1, 2, 3, 1 / 2, 1 / 3]

def classify_match(period, catalog_period, tol=0.01):
    """'match' if within 1% of the catalog period, 'alias' if within 1% of
    a simple multiple or fraction of it, otherwise 'miss'."""
    if not (np.isfinite(period) and np.isfinite(catalog_period)):
        return "n/a"
    for r in ALIAS_RATIOS:
        if abs(period / (catalog_period * r) - 1) < tol:
            return "match" if r == 1 else "alias"
    return "miss"

def evaluate(results, min_snr=7.0):
    """Score the results against the catalog.

    Adds two columns:
      period_match - match / alias / miss, based on period alone
      outcome      - same, but detections with SNR below `min_snr` become
                     'below_snr' (too weak to count, even if the period is right)

    Returns (successful_rows, summary_table_of_outcomes).
    """
    ok = results[results["status"] == "ok"].copy()
    ok["period_match"] = [classify_match(p, pc) for p, pc
                          in zip(ok["period_days"], ok["catalog_period"])]
    ok["outcome"] = np.where(ok["snr"] >= min_snr, ok["period_match"], "below_snr")
    table = pd.crosstab(ok["label"], ok["outcome"], margins=True)
    return ok, table
