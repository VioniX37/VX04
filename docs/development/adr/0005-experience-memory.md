# ADR 0005: Experience memory via pipeline hooks

- Status: Accepted
- Date: 2026-10-01

## Context

Every run discovers which model families work on which kinds of data, and which code errors occur and how they were fixed. The paper retrieves only static knowledge, so each run starts from scratch.

## Decision

- After each run, store an `ExperienceRecord` with dataset meta-features, the best plan and its test score, every candidate's predicted and observed scores, and error→fix pairs.
- Before planning, recall the `MEMORY_K` nearest past runs of the same task type (weighted meta-feature distance) as knowledge items with structured data.
- Offer past fixes to the Operation Agent when a failure's normalised error signature matches.
- Implement memory as a built-in `PipelineHooks` extension plus a `Retriever`, so the core pipeline is unchanged and memory can be switched off (`MEMORY_ENABLED=false`) for ablations.
- Provide a leave-one-dataset-out guard (`MEMORY_EXCLUDE_SAME_DATASET`) for evaluation.

## Alternatives considered

- *Embedding-based similarity of prompts or profiles.* Costs extra calls and is less interpretable; meta-features are free and explainable. Embeddings can be added as a re-ranker later.
- *Fine-tuning on past runs.* Not possible on the free tier, and opaque.

## Consequences

- Later runs can converge faster (fewer revisions and calls) on related data, which is what RQ3 measures.
- Memory is per workspace, so benchmark variants stay isolated.
- Bad past outcomes can mislead. Records carry observed scores and failures so the planner can weigh them, and memory never overrides grounded selection.
