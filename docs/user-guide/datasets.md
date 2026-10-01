# Datasets

## Formats

CSV, TSV, Parquet, JSON Lines (`.jsonl`, `.ndjson`) and JSON are accepted, and so are gzip-compressed versions of these (`.csv.gz`). Every dataset is converted **once**, at registration, into a single zstd-compressed Parquet file. The pipeline only ever reads that file.

## Three ways to add data

| Method | Best for | Limit |
|---|---|---|
| **Upload** (UI or `POST /api/datasets`) | Files on your computer | `MAX_UPLOAD_MB` (default 5 GB) |
| **Register by path** (UI tab, `automl-agent ingest --data PATH`) | Files already on the server: Kaggle/Colab inputs, mounted drives | Disk space; disable with `ALLOW_PATH_REGISTRATION=false` on shared servers |
| **Register by URL** | Public datasets (`http(s)://`) | 4 × `MAX_UPLOAD_MB` |

!!! tip "Large files: register by path"
    A browser upload is buffered in `<workspace>/tmp` and then copied into the dataset folder, so it temporarily needs about **twice the file's size** in free space on the workspace drive. If space runs out, the API answers `507` with a clear message. For multi-gigabyte files, registering by path is faster and needs no extra copy.

## Profiling

Registration also profiles the data. The agents only ever see this profile, never raw rows beyond three sample values per column.

- Row count, column count, size on disk and an in-memory estimate.
- Per column: type, detected *kind* (numeric, categorical, text, datetime, identifier, boolean), distinct and missing counts, samples, and value shares for low-cardinality columns.
- A guessed target column.
- A **scale tier**: `small` (< 100k rows), `medium` (< 2M) or `large`. The tier decides which model families are allowed and how grounding is scheduled.

Above 2M rows, distinct counts are HyperLogLog approximations. Text and identifier detection use an evenly strided 100k-row sample.

## Splits

When a run knows its target, it creates a fixed split for that (dataset, target) pair:

| Split | Share | Used for |
|---|---|---|
| train | 70% | Training (and nested subsamples during grounding) |
| valid | 15% | Early stopping, grounding scores |
| test | 15% | **Only** the final reported score |

Classification splits are stratified. The same split is reused by every plan, fidelity, baseline and repeated run, which makes all numbers directly comparable. Rows with a missing target are excluded. Shares and seed are configurable (`SPLIT_VALID_FRACTION`, `SPLIT_TEST_FRACTION`, `SPLIT_SEED`).

## Where data lives

```text
backend/workspace/datasets/<dataset id>/
  data.parquet                     # the converted dataset
  splits/<target>-<seed>.parquet   # __split (0/1/2) and __r (uniform draw) per row
```
