# Experience memory

*Contribution B.* The pipeline learns from its own history. Every run leaves a record of what worked and what failed; runs on similar datasets recall it as planning knowledge.

The paper's own implementation already feeds "past experience cases and knowledge written by an human expert" into planning. Those are static. Here, experience is **accumulated automatically**, it is **grounded** (observed scores, not predictions), and similarity is computed from dataset statistics.

## What is stored

After every run with a valid task specification, the built-in `MemoryHooks` extension writes an `ExperienceRecord`:

| Field | Content |
|---|---|
| `meta` | Dataset meta-features (below) |
| `dataset_fingerprint` | Hash of the schema and row count, which identifies the same dataset across registrations |
| `best_plan`, `best_score`, `target_met` | The selected plan, with its test score |
| `plans` | Every candidate: model family, predicted score, observed score at its highest rung, success |
| `fixes` | Error → fix pairs from the Operation Agent's debug loop |

## Similarity

Meta-features are computed from the profile and task only, never from raw data:

| Feature | Definition |
|---|---|
| `log_rows`, `log_cols` | $\log_{10}$ of rows and columns |
| `frac_numeric`, `frac_categorical`, `frac_text` | Share of feature columns of each kind |
| `frac_missing` | Missing cells / all cells |
| `log_classes` | $\log_2$ of the number of classes (0 for regression) |
| `log_imbalance` | $\log_{10}$ of majority share / minority share |

Distance is a weighted Euclidean distance over these features. Only records of the **same task type** are candidates, and the `MEMORY_K` nearest are recalled.

## How recall is used

1. **Planning.** Each recalled run becomes a knowledge item (`source: memory:<run id>`) describing the best plan, its hyperparameters and test score, the observed scores of the other candidates, and any families that failed to run. The planner prompt tells the Manager to prefer what worked and avoid what failed. The item also carries structured data (`best_model_family`, ...), so tools can act on it.
2. **Debugging.** Before the run, stored fixes are loaded. When a script fails, its error is reduced to a normalised signature, for example `KeyError: '…'` with data-specific values stripped. Past fixes with the same signature are shown to the Operation Agent.

## Avoiding leakage in evaluation

When a run is evaluated, it must not recall its own dataset. With `MEMORY_EXCLUDE_SAME_DATASET=true`, records with the same fingerprint are ignored (a leave-one-dataset-out protocol). The benchmark configs enable this. In normal use it is off, so repeated work on the same dataset benefits from memory.

## Configuration

| Setting | Default | Effect |
|---|---|---|
| `MEMORY_ENABLED` | `true` | Store and recall experience |
| `MEMORY_K` | 3 | Past runs recalled per task |
| `MEMORY_EXCLUDE_SAME_DATASET` | `false` | Leave-one-dataset-out guard |

Records live in the workspace database, so each workspace (and each benchmark variant) has its own memory.
