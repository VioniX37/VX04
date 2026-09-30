# ADR 0003: Grounded verification by successive halving

- Status: Accepted
- Date: 2026-10-01

## Context

The paper selects which plan to implement from the Model Agent's *predicted* scores. These are never validated, and their quality depends on the backbone LLM. On large data, fully training every candidate is too expensive.

## Decision

Add a `ground` stage that runs **successive halving over data fidelity**. Surviving plans train on nested subsamples (`GROUNDING_MIN_ROWS` × `GROUNDING_GROWTH`^i rows), are scored on a capped validation slice, and the best 1/`GROUNDING_ETA` advance. The selected plan is the best observed one. Rungs that would exceed the remaining wall-clock budget are skipped. Every prediction and observation is persisted for calibration analysis. The paper's behaviour stays available as `VERIFICATION_MODE=pseudo`.

## Alternatives considered

- *Full training of every plan.* Exact, but its cost is the number of plans × full training.
- *LLM-judged verification (paper).* Cheap, but unmeasured and backbone-dependent.
- *Bayesian optimisation over plans.* Heavier machinery; plans are few and discrete, so successive halving is enough and easier to explain.
- *Time-based fidelity (train for t seconds).* Less reproducible than row counts.

## Consequences

- Plan selection is robust to poorly calibrated LLM predictions, and the calibration itself becomes measurable (RQ1).
- Runs cost extra training time, bounded by the schedule and the budget; small datasets get a single full-data rung.
- Grounding runs use the rendered templates, with no LLM code editing, so they are cheap and reliable. LLM-edited code is used only for the final implementation.
