# Agents

Each agent is a role prompt (`backend/src/automl_agent/prompts/*.md`) plus a Pydantic output schema. Agents call Gemini through the [LLM router](llm-layer.md), which assigns them the *smart* or the *fast* model. Their inputs travel as a tagged JSON context block, so every call is structured, validated and reproducible.

| Agent | Class | Model role | Input | Output schema | Paper section |
|---|---|---|---|---|---|
| Prompt Agent | `PromptAgent` | fast | user request, dataset profile | `TaskSpec` | 3.1 |
| Agent Manager | `AgentManager` | smart | task spec, profile, knowledge, allowed models, budget, feedback | `PlanSet` | 3.2 |
| Data Agent | `DataAgent` | fast | task spec, profile, one plan, data sub-tasks | `DataAgentResult` | 3.3 |
| Model Agent | `ModelAgent` | fast | task spec, profile, one plan, model sub-tasks, allowed models | `ModelAgentResult` | 3.3 |
| Plan Analyst | `PlanAnalyst` | fast | as Data + Model (used when `AGENT_FUSION=true`) | `PlanAnalysis` | — (ours) |
| Operation Agent | `OperationAgent` | smart | task spec, plan, agent reports, working base script, errors, past fixes | `CodeDraft` | 3.5 |

## Prompt Agent

Parses the request into a `TaskSpec`:

- task type, target and text columns, columns to drop, metric and optional metric target;
- time budget, domain and the user's apparent expertise;
- a list of **assumptions** it had to make.

Request verification checks the spec against the dataset profile. If the spec is rejected, the issues are sent back to the Prompt Agent once for repair.

## Agent Manager

Orchestrates the pipeline (see [Pipeline stages](pipeline-stages.md)). As planner, it proposes `N_PLANS` meaningfully different end-to-end plans. Each plan names a model family from the scale-filtered registry, ordered preprocessing steps, hyperparameters and a rationale that cites the retrieved knowledge. Revisions receive feedback that includes predicted *and* observed scores from earlier rounds.

## Data and Model agents

The paper's **pseudo-execution**: each agent reasons about its decomposed sub-tasks without running code. The Data Agent refines preprocessing steps and lists data risks. The Model Agent fixes the model family and hyperparameters (training-free HPO) and **predicts** the hold-out score and training time. The prediction is logged for every plan, which is how we measure its calibration (RQ1).

## Operation Agent

Receives a **working base script** rendered from a template for the chosen model family, and returns a complete script that applies the plan. If the script fails, it gets the error, plus any matching fixes from [experience memory](experience-memory.md), and repairs it, up to `MAX_DEBUG_ATTEMPTS` times. If every edited version fails, the unmodified template is run as a safety net ([ADR 0004](../development/adr/0004-template-grounded-codegen.md)).

## Adding an agent

1. Add a role prompt in `prompts/` with the documentation header (role, model role, inputs, output schema).
2. Subclass `BaseAgent`, setting `name`, `prompt_name` and `model_role`, and call `self.ask_json(stage, task, context, Schema)`.
3. Add a handler `_answer_<Schema>` to `llm/fake.py` so the agent works offline and in tests.
