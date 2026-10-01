# ADR 0002: Parquet ingest and fixed, nested splits

- Status: Accepted
- Date: 2026-10-01

## Context

The system must handle datasets of up to ~5 GB / ~10M rows on laptops and notebook VMs. The first version read CSVs with pandas on every run, and each generated script chose its own random split. The first approach does not scale. The second makes scores from different plans, runs and baselines incomparable, and lets agents see test data.

## Decision

- Convert every dataset **once**, at registration, into zstd Parquet using Polars' streaming engine. All later reads (profiling, splits, training) use the Parquet file.
- Profile lazily: one pass for counts and nulls, approximate distinct counts above 2M rows, and a strided sample for type detection.
- For each (dataset, target), write a split file aligned row-by-row with the data. It holds `__split` (train/valid/test, stratified for classification) and `__r`, an independent uniform draw.
- Subsample for fidelity by filtering `__r < r / n`, so subsamples are **nested** across fidelities.
- Use the **test** split only for final scores.

## Alternatives considered

- *Split inside each script.* Simple, but plans and baselines are then scored on different rows, and the test data leaks into selection.
- *Store the split column inside the data file.* The right split depends on the target, which is known only after parsing.
- *DuckDB instead of Polars.* Comparable for our needs; Polars also offers convenient LazyFrame → pandas conversion for training.

## Consequences

- All comparisons (plans, fidelities, variants, baselines, seeds) use identical rows.
- Registration of multi-GB files takes seconds to minutes, once.
- Templates depend on the split-file contract; generated code must keep reading it.
