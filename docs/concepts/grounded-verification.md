# Grounded verification

*Contribution A.* This replaces the paper's execution verification, which ranks plans on LLM-predicted scores, with real, budgeted measurements.

## The problem

In AutoML-Agent, the Data and Model agents *pseudo-execute* each candidate plan: the Model Agent predicts the score a plan would reach, and the Manager keeps the most promising plan. This saves compute, but the predictions are never checked. Nothing guarantees that the plan with the highest predicted score is actually the best. The paper also reports that quality collapses with weaker backbone LLMs, exactly the setting where such predictions are least reliable.

## The method

We keep the agents' analysis, which still shapes preprocessing and hyperparameters, but select plans by running them. Training cost grows with data size, so we use **successive halving over data fidelity**. Every surviving plan trains on a nested random subsample of the training split and is scored on a capped validation slice. The weakest plans are then dropped and the subsample grows.

Let $P$ be the number of plans, $n$ the training-set size, $r_0$ = `GROUNDING_MIN_ROWS`, $g$ = `GROUNDING_GROWTH` and $\eta$ = `GROUNDING_ETA`. Rung $i$ trains on

$$ r_i = \min(r_0 \, g^{\,i},\; n) $$

rows and keeps the best $\lceil P_i / \eta \rceil$ of its $P_i$ plans. It stops when one plan remains or the full training set is reached. With the defaults ($r_0$ = 20k, $g$ = 4, $\eta$ = 2) and three plans on 3.5M training rows, the schedule is:

| Rung | Rows | Plans trained |
|---|---|---|
| 0 | 20,000 | 3 |
| 1 | 80,000 | 2 |
| final (Operation Agent) | 3,500,000 | 1 |

Datasets with at most $r_0$ training rows get a single full-data rung, so every plan is simply evaluated for real.

### Why nested subsamples

The split file stores an independent uniform draw $u \in [0,1)$ per row. A fidelity of $r$ rows keeps the training rows with $u < r / n_\text{train}$. Every smaller subsample is therefore a subset of every larger one, so a plan's score at rung $i+1$ is measured on a superset of the data it saw at rung $i$. Comparisons between plans always use identical rows.

### Budget awareness

Before each rung after the first, the Manager estimates its cost as

$$ \hat{t}_{i} = \bar{t}_{i-1} \cdot \frac{r_i}{r_{i-1}} \cdot P_i $$

(mean seconds per plan at the previous rung × row growth × plans). If it exceeds the remaining `BUDGET_WALL_S`, grounding stops and ranks on what has been observed. Each grounding run's timeout is also capped by the remaining time.

## What is logged

Every (plan, fidelity) pair is stored as a `PlanObservation`. The selected plan's final full-data run is stored too, scored on the test split:

| Field | Meaning |
|---|---|
| `predicted_score` | Model Agent's pseudo-execution prediction |
| `observed_score` | Measured validation score (grounding) or test score (`final=true`) |
| `fidelity_rows` | Training rows actually used |
| `duration_s`, `ok` | Cost and success of the run |

This log is the data for **RQ1** (how well pseudo-execution is calibrated) and for the calibration figure. See [Evaluation protocol](../research/evaluation-protocol.md).

## Example

Run on the 5M-row synthetic conversion dataset (ROC AUC). The pseudo-execution prior ranked LightGBM first; real runs disagreed:

| Plan | Predicted | 20k rows | 80k rows |
|---|---|---|---|
| LightGBM | **0.890** | 0.728 | 0.748 |
| XGBoost | 0.885 | 0.716 | eliminated |
| HistGradientBoosting | 0.880 | **0.750** | **0.765** |

The selected HistGradientBoosting plan reached **0.7778** test ROC AUC on the full 3.5M training rows, versus 0.7775 for the LightGBM plan the pseudo-ranking would have picked. The whole run took 56 s instead of 84 s.

!!! note
    These numbers come from the offline `fake` backend, whose predictions are fixed priors. They illustrate the mechanism, not Gemini's calibration. Measure that with the benchmark harness.

## Configuration

| Setting | Default | Effect |
|---|---|---|
| `VERIFICATION_MODE` | `grounded` | `pseudo` reproduces the paper (no real runs before selection) |
| `GROUNDING_MIN_ROWS` | 20000 | Rows at the first rung |
| `GROUNDING_GROWTH` | 4 | Row multiplier between rungs |
| `GROUNDING_ETA` | 2 | Keep the best $1/\eta$ each rung |
| `GROUNDING_VALID_ROWS` | 100000 | Validation rows used to score grounding runs |
