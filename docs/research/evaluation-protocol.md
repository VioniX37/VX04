# Evaluation protocol

## Metrics

**Success rate (SR).** This follows the paper's graded scheme, adapted to a pipeline that ends in a trained, verified model:

| SR | Condition |
|---|---|
| 0 | No working model |
| 0.5 | A working model, but a stated constraint was missed on the test split |
| 1 | A working model and every stated constraint met (for constraint-free prompts, any working model) |

**Normalized performance score (NPS).** Bounded "higher is better" metrics (accuracy, F1, ROC AUC) are used directly. A loss $s$ maps to $1/(1+s)$. For regression we use RMSLE whenever the target is non-negative, as the paper does.

**Comprehensive score.** $\text{CS} = 0.5 \cdot \text{SR} + 0.5 \cdot \text{NPS}$.

**Calibration (RQ1).** Within each planning round, take the grounding rung where *all* candidate plans ran on identical rows (the first rung):

- **Spearman ρ** between predicted and observed scores (rounds with at least 3 plans);
- **top-1 hit rate**, the share of rounds where the predicted-best plan is the observed-best;
- **MAE** $\lvert \hat{s} - s \rvert$ for bounded metrics.

**Cost.** Wall-clock seconds, Gemini calls, cache hits and tokens per run. **Spec accuracy**: the share of runs where the Prompt Agent identified the correct target and task type.

## Protocol

1. **Fixed splits.** A stratified 70/15/15 train/valid/test split per (dataset, target), seed 42, shared by every variant, seed and baseline. Grounding uses train and valid; only final scores use test.
2. **Variants run in isolated workspaces.** Each variant × seed has its own database, so memory accumulates only within that variant, in the configured task order.
3. **Leave-one-dataset-out memory.** `MEMORY_EXCLUDE_SAME_DATASET=true`: a run never recalls runs on its own dataset.
4. **Seeds.** Three seeds per condition. The seed selects the LLM cache namespace, so each seed gets fresh sampled responses while re-runs of the same seed are exactly reproducible.
5. **Equal budgets.** Constrained comparisons set `BUDGET_WALL_S`; the Optuna baseline receives the same wall-clock budget.
6. **Shared cache across variants.** Identical requests across variants (for example, the parse step) are answered once per seed, which saves quota without changing results.

## Running

```bash
cd backend
# Smoke test (offline, ~2 minutes)
LLM_PROVIDER=fake python -m evaluation.run_benchmark --config evaluation/configs/smoke.json

# Paper datasets (Gemini); fetch data first (needs the Kaggle CLI + accepted competition rules)
python -m evaluation.datasets.fetch --group paper
python -m evaluation.run_benchmark --config evaluation/configs/paper_tabular.json

# Scale study
python ../data/samples/generate_large.py --rows 5000000
python -m evaluation.datasets.fetch --group large
python -m evaluation.run_benchmark --config evaluation/configs/large.json

# Tables and figures (also copied into the paper and the docs)
python -m evaluation.analysis ../evaluation_results/<run dir> \
  --figures ../paper/figures --markdown ../docs/research/results.md
```

Useful options: `--variants grounded,full`, `--tasks banana_quality`, `--set BUDGET_WALL_S=600`, `--no-baselines`, `--isolated-cache`.

## Outputs

| File | Content |
|---|---|
| `results.csv` | One row per run: variant, task, prompt kind, seed, SR, NPS, CS, score, model, spec correctness, revisions, stop reason, wall time, calls, cache hits, tokens, memory hits |
| `observations.csv` | Every (plan, fidelity) predicted-vs-observed pair, and final test runs |
| `summary.json` | Means per variant and baseline, plus calibration |
| `table_main.md`, `table_calibration.md` | Paper-ready tables |
| `figures/` | `fig_cs_by_variant`, `fig_calibration`, `fig_memory_curve` (PDF and PNG) |

## Quota planning (free tier)

A full `paper_tabular` run is 4 datasets × 2 prompts × 5 variants × 3 seeds = 120 pipeline runs, at roughly 10–15 calls each. Spread it over several days, start with one seed, use `AGENT_FUSION=true`, and rely on the cache: an interrupted benchmark resumes at no cost for already-answered requests.
