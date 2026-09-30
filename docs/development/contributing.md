# Contributing

## Workflow

1. Create a branch from `main`: `feat/<topic>`, `fix/<topic>` or `docs/<topic>`.
2. Make focused commits using [Conventional Commits](https://www.conventionalcommits.org/): `feat(scope): ...`, `fix(scope): ...`, `docs: ...`, `test: ...`, `refactor: ...`, `style: ...`, `chore: ...`. Common scopes: `llm`, `data`, `verification`, `memory`, `evaluation`, `frontend`, `docs`.
3. Run the checks below; CI runs the same ones.
4. Open a pull request using the template. Describe the change, how it was tested and any documentation updated.
5. Record user-visible changes in `CHANGELOG.md`, and write an [ADR](adr/index.md) for decisions that shape the architecture.

## Checks

```bash
# Backend
cd backend
ruff check . && ruff format --check .
pytest
python scripts/gen_config_reference.py --check

# Frontend
cd frontend
npm run lint && npm run build

# Documentation (from the repository root)
mkdocs build --strict
```

## Code style

**Python**

- Ruff enforces formatting, imports, modern syntax and Google-style docstrings on every public module, class and function (`D` rules; tests are exempt).
- Type-annotate public functions. Agents exchange Pydantic models, not dicts.
- Put tunable values in `config.py`. A test fails if a setting is missing from the configuration reference; regenerate it with `python scripts/gen_config_reference.py`.

**TypeScript**

- ESLint with the Next.js config. Add TSDoc comments to exported components and library functions.
- Mirror backend schemas in `frontend/src/lib/types.ts` whenever they change.

**Prompts**

- Every file in `prompts/` starts with a header comment stating the role, model role, input context keys and output schema.

## Adding features

| To add... | Touch |
|---|---|
| A model family | `execution/model_registry.py` + the template's `build_model` + a fake prior in `llm/fake.py` |
| A setting | `config.py` (with a `description`), then regenerate the configuration reference |
| A knowledge source | Implement the `Retriever` protocol and add it in `AgentManager.default_retrievers` |
| A pipeline extension | Subclass `PipelineHooks` and call `register_hooks(...)` |
| A benchmark dataset | `evaluation/datasets/fetch.py` + a config entry with the ground-truth `truth` spec |
