# Human in the loop

*Issue #4.* A fully automatic pipeline is the right default for benchmarks, but a practitioner often knows things the agents don't. A model family might be impossible to deploy, a feature might be known to leak, or the time might be better spent elsewhere. Every run therefore has an **approval mode**, and every running run can be **cancelled**.

## Approval modes

| Mode | Behaviour |
|---|---|
| `auto` (default) | No pauses. CLI and benchmark runs behave exactly as before |
| `plans` | After grounded verification ranks the plans, the run pauses as `awaiting_input` and shows the ranked plans with their **observed** scores |
| `plans+code` | Pauses for the plan *and* again for the generated training script, shown as a diff against the template |

The mode is chosen in the new-run form (`RunCreate.approval`) and stored on the `RunRecord`.

## Deciding

A paused run waits for `POST /api/runs/{id}/approve` with a `PlanApprovalRequest`:

| `action` | Effect |
|---|---|
| `approve` | Continue with the top-ranked plan |
| `pick` + `plan_id` | Continue with another plan from the ranking |
| `edit` + `edited_plan` | Change fields of the top plan, for example `model_family` or `hyperparameters`, and continue |
| any + `edited_code` | At the code step, replace the script with the edited one |

An edited model family is mapped to a supported one before training (`AgentManager._validated_choice`). An unknown name such as `"LightGBM-DART"` becomes `lightgbm`, and a warning event is emitted. A choice that differs from the agent's sets `human_override` on the run and on its experience-memory record, so later analyses can tell human decisions from automatic ones.

If nobody answers within `APPROVAL_TIMEOUT_S` (3600 s by default; 0 waits indefinitely), the run continues with the top-ranked plan and records that the approval timed out. An unattended run can never hang forever.

## How pausing works

`extensions/approval.py:ApprovalHooks` implements `on_plans_ranked` and `on_code_generated`:

1. It writes `runs/<id>/pause_state.json` with the step, the ranked plans (or the code) and the default choice.
2. It sets the run to `awaiting_input` and emits an event carrying the same payload, which the UI's `ApprovalCard` renders.
3. It awaits an asyncio future registered under the run id. `approve_run` resolves that future with the decision.

### Surviving a restart

A paused run holds no process resources, so a server restart must not lose it. `recover_interrupted_runs` marks crashed `running` runs as failed but leaves `awaiting_input` runs alone. When a decision arrives for a run that has no in-memory future, `approve_run` does three things:

1. It writes the decision into `pause_state.json`.
2. It sets the run back to `running`.
3. It re-executes the pipeline.

Re-planning after a restart can produce different plans, or the same plans in a different order, even with the response cache. So when the hook reaches the pause point, it applies the decision to the **plans saved in `pause_state.json`**, the ones the user actually reviewed. The pause is not repeated.

## Cancelling

`POST /api/runs/{id}/cancel` works for running and paused runs:

1. It kills every sandbox process tree started for the run (`execution/sandbox.py:kill_run_processes`). Training scripts are tracked per run id when they start.
2. It cancels any pending approval future.
3. It cancels the pipeline task, which stops further LLM calls at the next `await`.
4. In the task's `CancelledError` handler, it keeps the partial plan observations, records LLM usage, and sets the status to `cancelled` with a `run_cancelled` event.

The run list and run view show `cancelled` runs as finished.
