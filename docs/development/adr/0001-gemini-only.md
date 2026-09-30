# ADR 0001: Gemini-only LLM layer with role routing

- Status: Accepted
- Date: 2026-10-01

## Context

The first version supported several providers (OpenAI, Anthropic, Gemini, Ollama, and others) through a common interface. The project will use **Gemini exclusively**, on a **free AI Studio key** with strict per-minute limits. The paper runs all agents on one model (GPT-4o), although the agents' workloads differ a lot: planning and code generation need strong reasoning, while request parsing and per-plan analysis are high-volume and simpler.

## Decision

- Keep the provider-neutral `LLMClient` interface, but ship only two implementations: `GeminiClient` (google-genai SDK) and `FakeLLM` (offline tests). Remove the other adapters and their dependencies.
- Route agents by **role**: `smart` (Manager, Operation Agent) and `fast` (Prompt, Data, Model agents, Plan Analyst), each with its own model id and rate limit.
- Use Gemini's native **JSON-schema output** for every agent call, with a prompt-described fallback.
- Make the free tier workable: per-model sliding-window rate limits, a global concurrency cap, backoff that honours `retryDelay`, an on-disk response cache, and an optional fused Data+Model call.
- Use Gemini's **Google Search grounding** for web knowledge instead of separate search APIs.

## Alternatives considered

- *Keep all providers.* More code to maintain and test, and none of it would be used.
- *One model for all roles.* Simpler, but it wastes quota on easy calls or under-powers planning.
- *No cache.* Experiments would be irreproducible, and every re-run would spend quota.

## Consequences

- The interface still allows adding a provider later with one class.
- Model ids change as Gemini evolves. Ids are settings, validated at startup, and recorded per run.
- Cached answers make experiments reproducible; the cache namespace (seed) controls when fresh samples are drawn.
