"""
run.py - command-line entry point for the pipeline.

This is what Docker and AWS Batch runs. Examples:

    # a few named stars
    python -m src.run --targets "WASP-18" "pi Men" --name quick_test

    # every star in a targets file (a CSV saved from select_sample)
    python -m src.run --targets-file data/targets_train_v1.csv --name train_v1

    # in the cloud: read the target list from S3 and upload results to S3
    python -m src.run --targets-file s3://my-bucket/targets/targets_train_v1.csv \
        --name train_v1 --s3-output s3://my-bucket/results

    # one chunk of a large file, for parallel jobs. On AWS Batch each copy
    # of the job gets its own AWS_BATCH_JOB_ARRAY_INDEX (0, 1, 2, ...), so
    # 20 copies with --chunk-size 50 cover 1,000 stars between them.
    python -m src.run --targets-file data/targets_train_v1.csv --name train_v1 --chunk-size 50
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .batch import DATA_DIR, run_batch


def _upload_dir(local_dir, s3_uri):
    """Copy every file under local_dir to s3_uri, keeping the folder layout.

    Uses s3fs, which is already installed because lightkurve depends on it.
    Inside AWS, credentials come automatically from the job's permissions.
    """
    import s3fs

    fs = s3fs.S3FileSystem()
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    for p in files:
        fs.put_file(str(p), f"{s3_uri}/{p.relative_to(local_dir).as_posix()}")
    print(f"Uploaded {len(files)} file(s) to {s3_uri}", flush=True)

# Columns run_batch reads from each row. Ad-hoc targets won't have catalog
# information, so any missing ones are filled with blanks.
_EXPECTED = ["toi", "pl_orbper", "pl_trandep", "st_tmag", "st_rad", "st_logg", "st_teff"]


def _load_targets(args):
    if args.targets_file:
        df = pd.read_csv(args.targets_file)
    else:
        df = pd.DataFrame({"target": args.targets})
    for col in _EXPECTED:
        if col not in df.columns:
            df[col] = np.nan
    if "label" not in df.columns:
        df["label"] = "unknown"
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="TESS transit search pipeline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--targets", nargs="+", help="star names or 'TIC <id>' strings")
    src.add_argument("--targets-file", help="CSV with a 'target' column (e.g. from select_sample)")
    p.add_argument("--name", default="run", help="output folder name under data/")
    p.add_argument("--max-sectors", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=0,
                   help="process only one chunk of this many targets (0 = all)")
    p.add_argument("--chunk-index", type=int, default=None,
                   help="which chunk; defaults to $AWS_BATCH_JOB_ARRAY_INDEX, else 0")
    p.add_argument("--no-plots", action="store_true", help="skip saving diagnostic plots")
    p.add_argument("--s3-output", default=None,
                   help="if set, upload this run's results folder to s3://bucket/prefix when done")
    args = p.parse_args(argv)

    df = _load_targets(args)
    name = args.name

    if args.chunk_size > 0:
        idx = args.chunk_index
        if idx is None:
            idx = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))
        df = df.iloc[idx * args.chunk_size:(idx + 1) * args.chunk_size]
        name = f"{name}/chunk_{idx:04d}"
        if df.empty:
            print(f"Chunk {idx} is empty - nothing to do.")
            return 0

    print(f"Processing {len(df)} target(s) -> {DATA_DIR / name}", flush=True)
    run_batch(df, name=name, max_sectors=args.max_sectors, save_plots=not args.no_plots)

    if args.s3_output:
        _upload_dir(DATA_DIR / name, f"{args.s3_output.rstrip('/')}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
