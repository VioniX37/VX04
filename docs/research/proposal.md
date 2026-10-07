# Research proposal

**Working title:** *Grounded AutoML-Agent: Budget-Aware Multi-Fidelity Verification and Experience Memory for Scalable LLM-Driven AutoML* (final name: **GroundML**)

## Motivation

AutoML-Agent shows that a team of LLM agents can automate the full ML pipeline. Its efficiency rests on **pseudo-execution**: before any code runs, the Model Agent predicts how well each candidate plan will perform, and the Manager keeps the most promising plan. Three observations motivate this work:

1. **The predictions are never checked.** No reported experiment measures whether predicted scores agree with reality. If they do not, plan selection is close to arbitrary among plausible plans.
2. **The approach is fragile to the backbone.** The authors report failures with smaller LLMs, which are precisely the models whose predictions should be least reliable. Cheap, free-tier models are what most practitioners and students can use.
3. **Scale is untested.** All benchmark datasets are small; the system ran on eight A100 GPUs. Real tabular problems often have millions of rows and must run on one CPU machine under a time budget.

## Research questions

| | Question | Measured by |
|---|---|---|
| **RQ1** | How well calibrated are pseudo-execution predictions? | Spearman ρ, top-1 hit rate and MAE between predicted and observed validation scores |
| **RQ2** | Does grounded multi-fidelity verification improve outcomes at equal cost, especially on large data? | SR, NPS, CS, wall time and LLM calls: `grounded` vs `paper`, at equal `BUDGET_WALL_S` |
| **RQ3** | Does experience memory reduce cost and raise success over a sequence of tasks? | LLM calls, revisions and CS over task order: `memory` vs `paper`, `full` vs `grounded` (leave-one-dataset-out) |
| **RQ4** | With grounding and memory, can a free-tier Flash model match the paper's GPT-4o results? | CS of `full` (Gemini Flash) vs the numbers reported in the paper, plus our baselines |

## Hypotheses

- **H1.** Pseudo-execution predictions correlate weakly with observed scores (ρ < 0.5), and the predicted-best plan is frequently not the observed-best.
- **H2.** Grounded verification increases NPS and CS over pseudo verification. The gain grows with dataset size and with weaker backbones, at a bounded extra wall-clock cost.
- **H3.** Memory reduces LLM calls and revisions and increases SR for later tasks in a sequence of related datasets.
- **H4.** The combined system with a Flash-class model reaches CS within a small margin of the published GPT-4o results on the overlapping tabular and text tasks.

## Method

- **Grounded verification.** Successive halving over nested data subsamples, budget-aware rung scheduling, selection on observed validation scores and final scoring on a held-out test split. See [Grounded verification](../concepts/grounded-verification.md).
- **Experience memory.** Meta-feature similarity search over past runs; recalled plans and observed scores feed retrieval-augmented planning; error→fix memory feeds debugging. See [Experience memory](../concepts/experience-memory.md).
- **Scalable execution substrate.** Parquet ingest, Polars profiling, fixed stratified splits, LightGBM/XGBoost with early stopping, out-of-core text models and a supervised sandbox. See [Large data](../concepts/large-data.md).

## Experimental design

| Factor | Levels |
|---|---|
| Verification | `pseudo` (paper-faithful), `grounded` |
| Memory | off, on (leave-one-dataset-out) |
| Agent calls | separate, fused (`AGENT_FUSION`) |
| Backbone | Gemini smart/fast pair; optionally one model for both roles |
| Seeds | 3 (LLM sampling via cache namespace; splits fixed) |
| Datasets | Paper tabular/text sets (Banana Quality, Software Defects, Crab Age, Ecommerce Text) and large sets (synthetic 5M, HIGGS 11M, NYC Taxi, Amazon Polarity) |
| Prompts | Constraint-free and constraint-aware, as in the paper |
| Baselines | Gemini zero-shot single script; Optuna + LightGBM at equal wall-clock budget |

Details, metrics and commands are in the [evaluation protocol](evaluation-protocol.md).

## Expected contributions

1. The first measurement of how well LLM pseudo-execution in agentic AutoML is calibrated.
2. A budget-aware grounded verification procedure that makes selection robust to the backbone.
3. An experience memory that turns every run into reusable, measured planning knowledge.
4. An open, reproducible implementation that runs on free-tier LLMs and large data on commodity hardware, with a web interface and an evaluation harness.

## Threats to validity

- **Model drift.** Gemini models change over time. Mitigated by recording model ids per run and caching all responses.
- **Dataset overlap with pre-training.** Public Kaggle data may be memorised. We report results on synthetic data too.
- **Comparison with published numbers (RQ4).** Different splits and hardware. We treat this as indicative and rely on our own baselines for controlled claims.
- **Heuristic fake backend.** Used only for tests. All reported results use Gemini.

## Future work

Human-in-the-loop plan approval in the UI, a deployment endpoint for trained models, additional modalities (time series, images), and LLM-judged request clarification with multi-turn dialogue.
