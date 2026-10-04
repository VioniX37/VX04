# Python API

Generated from the source docstrings. The most useful entry points for extending the system are the Manager, the extension hooks, the retriever protocol and the settings.

## Pipeline

::: automl_agent.agents.manager
    options:
      members: [AgentManager, PipelineResult]

::: automl_agent.agents.context

::: automl_agent.agents.budget

## Agents

::: automl_agent.agents.base

::: automl_agent.agents.prompt_agent

::: automl_agent.agents.data_agent

::: automl_agent.agents.model_agent

::: automl_agent.agents.plan_analyst

::: automl_agent.agents.operation_agent

## Extensions

::: automl_agent.extensions

## Verification

::: automl_agent.verification.request

::: automl_agent.verification.execution

::: automl_agent.verification.grounding

::: automl_agent.verification.implementation

## Experience memory

::: automl_agent.memory.meta_features

::: automl_agent.memory.store

::: automl_agent.memory.retriever

::: automl_agent.memory.hooks

## Planning

::: automl_agent.planning.retrieval

::: automl_agent.planning.gemini_search

::: automl_agent.planning.decomposition

## LLM layer

::: automl_agent.llm.base

::: automl_agent.llm.router

::: automl_agent.llm.gemini

::: automl_agent.llm.cache

::: automl_agent.llm.rate_limit

::: automl_agent.llm.factory

## Data tools

::: automl_agent.tools.ingest

::: automl_agent.tools.dataset_profiler

::: automl_agent.tools.splits

## Execution

::: automl_agent.execution.model_registry

::: automl_agent.execution.renderer

::: automl_agent.execution.sandbox

::: automl_agent.execution.inference

## Schemas

::: automl_agent.schemas.task_spec

::: automl_agent.schemas.dataset

::: automl_agent.schemas.plan

::: automl_agent.schemas.events

## Settings

::: automl_agent.config
    options:
      members: [Settings, get_settings]
