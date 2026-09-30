# ADR 0004: Template-grounded code generation

- Status: Accepted
- Date: 2026-10-01

## Context

The paper notes that smaller LLMs produce truncated or hallucinated code, and that skeleton scripts are needed. Free-tier models must still reach a working model reliably.

## Decision

- Plans must choose a model family from a registry. Each family is implemented in a per-task template with a fixed contract: read the Parquet data and split, train, and write `metrics.json` plus `model.joblib`.
- The Operation Agent receives the **rendered, working** template as a base and returns a full script that applies the plan. Failures are repaired in a debug loop.
- If every edited version fails, the unmodified template runs as a safety net. `CODEGEN_MODE=template` skips LLM editing entirely.
- The registry limits families by training-set size.

## Alternatives considered

- *Free-form code generation.* Maximum flexibility, but low reliability with weaker models.
- *Configuration-only (no LLM code).* Reliable, but loses the Operation Agent's ability to apply bespoke preprocessing.

## Consequences

- Near-100% success at producing *a* model, even with weak backbones.
- Adding a model family or modality is a local change: registry entry, template branch, and a fake-LLM prior.
- Evaluation reports the template-fallback rate so that its effect is transparent.
