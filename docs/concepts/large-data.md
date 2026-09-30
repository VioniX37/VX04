# Large data

The pipeline targets datasets of up to a few gigabytes, around 10M rows, on a single machine: a laptop, a Colab VM or a Kaggle notebook. Every stage is designed so that memory use stays bounded and runtime scales roughly linearly.

## End to end

| Stage | Technique |
|---|---|
| Upload | Streamed to disk in 16 MB chunks with a size cap; or registered by path/URL with no browser transfer |
| Ingest | Polars `scan_* → sink_parquet` (streaming engine), zstd compression, 100k-row row groups; `.gz` decompressed on the fly |
| Schema | Inference on the first 100k rows, retried with a full pass if a later row breaks it |
| Profiling | One lazy pass for counts and nulls; HyperLogLog distinct counts above 2M rows; strided 100k-row sample for text/ID detection |
| Splits | Only the target column is loaded; the split file is two tiny columns aligned with the data |
| Model choice | Registry removes families beyond their scale (`svm` ≤ 20k rows, `knn` ≤ 100k, random forests ≤ 1M, ...) |
| Grounding | Plans are compared on 20k → 80k → … rows; only the winner trains on everything |
| Training | Parquet → Arrow → pandas with `category` dtypes and float32 downcasting; LightGBM / XGBoost (hist) with native categoricals and early stopping on the validation split |
| Text | TF-IDF models up to 1M rows; beyond that, a stateless hashing vectorizer with `partial_fit` streams chunks from Parquet (out-of-core) |
| Sandbox | Process-tree memory watchdog (`EXEC_MAX_MEM_MB`, default 80% of RAM) and timeout; live progress lines |

## Measured scale test

Synthetic binary classification, 5,000,000 rows × 18 columns (12 numeric with missing values, 3 categoricals including one with 300 levels), on a laptop CPU with the offline LLM:

| Mode | Plans compared | Selected model | Test ROC AUC | Wall time | Peak RSS |
|---|---|---|---|---|---|
| Pseudo (paper) | none run | LightGBM | 0.7775 | 84 s | 2.0 GB |
| Grounded | 3 @ 20k, 2 @ 80k | HistGradientBoosting | 0.7778 | 56 s | ≈ 2 GB |

Reproduce it:

```bash
python data/samples/generate_large.py --rows 5000000
cd backend
LLM_PROVIDER=fake automl-agent --set CODEGEN_MODE=template run \
  --data ../data/samples/large_conversion.parquet \
  --prompt "Predict whether the customer converted. Optimise ROC AUC."
```

## Practical limits and tips

- **Memory.** Tabular training holds the training split in memory as compact pandas (roughly 0.4× the CSV size). Datasets far beyond RAM need the out-of-core text path or a bigger machine.
- **High-cardinality strings.** Tree boosters handle them natively. Linear models one-hot encode at most 30 levels per column. Drop pure identifiers via the request.
- **Time.** Set `BUDGET_WALL_S` so grounding sizes itself to the time you have. Raise `EXEC_TIMEOUT_S` for very large final fits.
- **Threads.** `EXEC_N_JOBS` controls CPU threads per training process.
