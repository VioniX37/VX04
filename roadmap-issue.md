# Roadmap: serving, human-in-the-loop, data audit, forecasting

**Labels:** `tracking`, `roadmap`

This issue tracks the four feature issues and the order they land in.

| Issue | Owner | Support |
|---|---|---|
| #3 Model serving | Aniketh | Jason (design review) |
| #4 Human-in-the-loop run control | Jason | — |
| #5 Data audit and trust report | Sajish | Pranav (method) |
| #6 Time-series forecasting | Pranav | Jason (implementation) |

## Why this order

- **Shared groundwork comes first.** #3, #5 and #6 all need the training preprocessing to be importable code instead of text rendered into the script. If each person refactors it separately, we get three conflicting versions.
- **#5's leakage checks come before #6's implementation.** Look-ahead leakage is the main way forecasting goes wrong, so the guard has to exist before the feature it guards.
- **#5's model card comes after #3's bundle**, because the card ships inside the bundle.
- **#4 is independent of the others** but touches `manager.py` and the run states that everyone builds on. Its schema changes land early and its UI lands later.

## Phase 0: groundwork

Goal: unblock everyone and agree on shared interfaces before parallel work starts.

- [ ] **Jason + Aniketh:** extract the preprocessing from `execution/templates/tabular.py` and `text_classification.py` into an importable module (for example `execution/preprocessing.py`). The rendered templates call it, and their output does not change. This is a pure refactor, and the existing tests must pass.
- [ ] **Jason:** open a short PR that adds the new `RunStatus` values (`awaiting_input`, `cancelled`), their migration and the new event types to `schemas/`, with no behaviour yet. This gives everyone the final schema.
- [ ] **Sajish:** add a synthetic dataset with a planted leaky column to `data/samples/`.
- [ ] **Pranav:** write a one-page design note for #6 covering the `TaskSpec` fields, the temporal split rules, the suffix-window rungs and the baselines. Post it on #6 for comments.

**Exit:** the refactor and schema PRs are merged, and the #6 design note is agreed.

## Phase 1: core backends

All four people work in parallel. Each piece ships in small PRs.

- [ ] **#4 (Jason):** `POST /runs/{id}/cancel`. It kills the sandbox process tree, stops LLM calls at the next await point and saves partial observations. Test it on Windows and Linux.
- [ ] **#3 (Aniketh):** the inference module, `POST /runs/{id}/predict` with schema validation, and `automl-agent predict` in the CLI.
- [ ] **#5A (Sajish):** the leakage checks (single-feature predictive power, ID-like columns, train/test duplicates) and the `AuditFinding` schema, running after `Prepare`.
- [ ] **#6 (Pranav):** the synthetic multi-store sales dataset, the choice of a public forecasting dataset, and its entry in the evaluation fetcher.

**Exit:** cancel works end to end; single predictions match the training script's predictions on the test split; the planted leak is flagged.

## Phase 2: features and UI

- [ ] **#4 (Jason):** `plans` approval mode, the fallback to `auto` on timeout, resuming after a backend restart, and the "Needs your input" UI.
- [ ] **#3 (Aniketh):** streaming batch scoring, the bundle export (`predict.py`, pinned `requirements.txt`, `schema.json`) and the "Use this model" panel.
- [ ] **#5A (Sajish):** the quality checks; feeding findings into `drop_columns`, the surfaced assumptions and planning knowledge; and the audit panel in the UI.
- [ ] **#6 (Pranav + Jason):** temporal splits in `tools/splits.py`, datetime and frequency detection in the profiler, suffix-window rungs in grounded verification, and a look-ahead leakage test. This starts once #5A's leakage checks are merged.

**Exit:** #3 and #4 can be closed, apart from #4's memory signal; the audit is visible in the UI and changes the task spec.

## Phase 3: second-stage features

- [ ] **#5B (Sajish, with Aniketh's help on the bundle):** the model card, built from feature importances, per-class metrics or residuals, and calibration. It renders in the UI and is included in the #3 bundle.
- [ ] **#6 (Pranav + Jason):** the forecasting template (seasonal-naive, ETS and LightGBM on lag features); the Prompt Agent extracting horizon and frequency; a KB entry; serving support for forecasting runs.
- [ ] **#4 (Jason):** record human overrides in experience memory, plus `plans+code` review mode if time allows.

**Exit:** a forecasting run works end to end from the UI and the CLI; every task type gets a model card.

## Phase 4: evaluation and release

- [ ] **Pranav:** ablation with the audit on and off, the forecasting benchmark against seasonal-naive, and an update to `docs/research/results.md` and the paper sections.
- [ ] **Everyone:** update the docs for their own feature (REST, CLI, event schema, user guide and the comparison table).
- [ ] **Jason:** integration pass. Run a full run of every task type with approval on and off, then serve and batch-score the result.
- [ ] Update `CHANGELOG.md` and tag a release.

## Rules to avoid merge pain

- **Files more than one feature touches**, each with one owner who reviews every change:
  - `agents/manager.py`: Jason (#4, #5, #6)
  - `schemas/task_spec.py`: Pranav (#5, #6)
  - `execution/preprocessing.py`: Aniketh (#3, #5B, #6)
  - frontend run view: whoever merges next rebases first
- **Small PRs, merged often.** Avoid long-lived branches. Branch off `main` and rebase often.
- **Defaults stay unchanged:** `approval=auto` and the audit can be turned off. The CLI and the evaluation harness must not change behaviour unless a flag is set.
- **Regular sync.** Each person reports what merged, what's blocked and the next PR. Move a task to another phase in this issue if it slips.
