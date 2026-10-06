# ADR 0006: Serving replays the training feature preparation

- Status: Accepted
- Date: 2026-10-05

## Context

Runs produced `model.joblib`, but nothing served it. The first serving implementation prepared features on its own: it rebuilt categories from each request, passed text models a DataFrame, and served the lexicographically last attempt. Predictions looked plausible but did not match what the model was trained and evaluated on.

## Decision

- Templates record their feature preparation in the bundle: feature order, training dtypes, category levels, ordinal-encoded columns, and excluded or unused columns.
- One standalone serving module (`execution/inference.py`) replays that preparation. The API, the CLI, the model card and the exported bundle all use it; the bundle ships the file itself.
- Inputs are validated strictly. Missing or unexpected columns and unconvertible values return 422 naming the column. Training columns the model did not use are accepted and ignored.
- The served model is the attempt the run selected (`metrics.artifact_dir`).
- Parity is tested: served predictions on the test split must reproduce the training script's test score.

## Alternatives considered

- *Import the training code into the API.* The templates are rendered scripts with an embedded `CONFIG`, not importable modules, and the bundle must run without the package.
- *Pickle a full sklearn pipeline including preparation.* Native-categorical models (LightGBM, XGBoost, HistGB) take pandas categoricals directly, so the preparation sits outside any pipeline.

## Consequences

- No training/serving skew for the supported task types, enforced by a test.
- Bundles from before this change load in a compatibility mode (object columns cast to category).
- Any new template must record its preparation in the bundle, or extend `prepare_features`.
