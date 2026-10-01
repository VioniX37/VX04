# Contributing

Thanks for helping improve Grounded AutoML-Agent.

This is proprietary software (see [LICENSE](LICENSE)). Contributions are accepted only from authorised contributors, and by contributing you agree that your contribution becomes part of the Software under the same all-rights-reserved terms.

The full guide, covering workflow, commit conventions, code style and how to add model families, settings, knowledge sources and extensions, is in the documentation: [docs/development/contributing.md](docs/development/contributing.md).

Quick checklist before opening a pull request:

```bash
cd backend && ruff check . && ruff format --check . && pytest && python scripts/gen_config_reference.py --check
cd frontend && npm run lint && npm run build
mkdocs build --strict        # from the repository root
```

- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat(memory): ...`, `fix(data): ...`, `docs: ...`).
- Update `CHANGELOG.md` for user-visible changes.
- Add or update an ADR in `docs/development/adr/` for architectural decisions.
