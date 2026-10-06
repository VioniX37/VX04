# Architecture decision records

Architecture Decision Records (ADRs) capture decisions that shape the system: the context, the options considered, what was chosen and the consequences. New ADRs are numbered sequentially and never rewritten; a later ADR supersedes an earlier one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-gemini-only.md) | Gemini-only LLM layer with role routing | Accepted |
| [0002](0002-parquet-ingest.md) | Parquet ingest and fixed, nested splits | Accepted |
| [0003](0003-grounded-verification.md) | Grounded verification by successive halving | Accepted |
| [0004](0004-template-grounded-codegen.md) | Template-grounded code generation | Accepted |
| [0005](0005-experience-memory.md) | Experience memory via pipeline hooks | Accepted |
| [0006](0006-serving-replays-training-preparation.md) | Serving replays the training feature preparation | Accepted |
| [0007](0007-forecasting-core.md) | A shared forecasting core with rolling-origin evaluation | Accepted |

Template:

```markdown
# ADR NNNN: Title

- Status: Proposed | Accepted | Superseded by ADR XXXX
- Date: YYYY-MM-DD

## Context
## Decision
## Alternatives considered
## Consequences
```
