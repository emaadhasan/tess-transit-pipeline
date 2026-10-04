"""
build_app_assets.py - prepares the small files the Streamlit app needs.

The pipeline's full results live in data/. This script copies just what the app needs into
app/assets/, which is committed:

    candidates.csv          the 403 ranked candidates (selected columns)
    plots/TIC_*.webp        each candidate's diagnostic plot, compressed
    feature_reference.json  typical feature values for planets vs false positives
    recovery_by_depth.csv   how often confirmed planets were recovered, by depth

Run it from the project root with the pipeline environment active:

    python scripts/build_app_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.batch import classify_match            
from src.features import add_derived_features   

DATA = ROOT / "data"
OUT = ROOT / "app" / "assets"     # not "data": .gitignore excludes every folder named data
PLOTS_OUT = OUT / "plots"

CANDIDATE_COLS = [
    "target", "toi", "score_physics", "score_all", "too_big", "weaker_than_training",
    "period_days", "duration_days", "depth_ppm", "snr", "snr_red", "n_transits", "sectors",
    "planet_radius_re", "st_rad", "st_tmag", "st_teff",
    "shape_ratio", "odd_even_sigma", "duty_cycle", "secondary_ratio", "red_noise_beta",
    "depth_chi2", "duration_ratio", "centroid_source_offset_px",
]

REFERENCE_FEATURES = [
    "shape_ratio", "odd_even_sigma", "duty_cycle", "secondary_ratio", "red_noise_beta",
    "depth_chi2", "duration_ratio", "planet_radius_re", "centroid_source_offset_px",
]


def build_candidates():
    rank = pd.read_csv(DATA / "candidate_ranking_v1.csv")
    if "too_big" not in rank.columns:
        rank["too_big"] = rank["planet_radius_re"] > 22
    cols = [c for c in CANDIDATE_COLS if c in rank.columns]
    rank[cols].to_csv(OUT / "candidates.csv", index=False)
    print(f"candidates.csv: {len(rank)} candidates")
    return rank


def build_plots(rank, width=1000, quality=80):
    """Shrink each candidate's plot and save it as WebP (about 5x smaller than PNG)."""
    PLOTS_OUT.mkdir(parents=True, exist_ok=True)
    done, missing = 0, []
    for target in rank["target"]:
        name = target.replace(" ", "_")
        src = DATA / "candidates_v1" / "plots" / f"{name}.png"
        if not src.exists():
            missing.append(target)
            continue
        img = Image.open(src).convert("RGB")
        if img.width > width:
            img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        img.save(PLOTS_OUT / f"{name}.webp", "WEBP", quality=quality)
        done += 1
    size_mb = sum(p.stat().st_size for p in PLOTS_OUT.glob("*.webp")) / 1e6
    print(f"plots: {done} saved ({size_mb:.1f} MB), {len(missing)} missing")


def build_training_reference():
    """Typical feature values for each class, and planet recovery by transit depth."""
    res = pd.read_csv(DATA / "train_v1" / "results.csv")
    df = add_derived_features(res[res["status"] == "ok"].copy())
    df["period_match"] = [classify_match(p, c) for p, c in zip(df["period_days"], df["catalog_period"])]

    # Same cleaning as notebook 05: planets whose catalog period was recovered, all false positives
    clean = df[(df["label"] == "false_positive") | (df["period_match"] == "match")]
    reference = {}
    for feat in REFERENCE_FEATURES:
        reference[feat] = {}
        for label, group in clean.groupby("label"):
            values = group[feat].replace([np.inf, -np.inf], np.nan).dropna()
            reference[feat][label] = {
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
            }
    (OUT / "feature_reference.json").write_text(json.dumps(reference, indent=2))
    print("feature_reference.json: written")

    planets = df[df["label"] == "planet"].copy()
    planets["recovered"] = (planets["period_match"] == "match") & (planets["snr"] >= 7)
    bins = [0, 300, 1000, 3000, 10000, np.inf]
    labels = ["under 300", "300 to 1,000", "1,000 to 3,000", "3,000 to 10,000", "over 10,000"]
    planets["depth_bin"] = pd.cut(planets["catalog_depth_ppm"], bins=bins, labels=labels)
    rec = (planets.groupby("depth_bin", observed=False)["recovered"]
           .agg(recovered="mean", planets="count").reset_index())
    rec.to_csv(OUT / "recovery_by_depth.csv", index=False)
    print("recovery_by_depth.csv: written")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ranking = build_candidates()
    build_plots(ranking)
    build_training_reference()
    print(f"\nDone. Assets are in {OUT.relative_to(ROOT)}")
