"""On-disk cache of LLM responses.

Makes experiments reproducible (a re-run replays identical answers) and saves
free-tier quota. Entries are keyed by a hash of everything that determines
the answer: model, messages, temperature, JSON mode and schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .base import LLMResponse, Message


class ResponseCache:
    """A directory of JSON files, one per cached response, sharded by key prefix."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(
        model: str,
        messages: list[Message],
        temperature: float,
        json_mode: bool,
        schema: dict[str, Any] | None,
        *,
        extra: str = "",
    ) -> str:
        """Return a stable hex digest identifying a request."""
        payload = json.dumps(
            {"m": model, "msg": messages, "t": temperature, "j": json_mode, "s": schema, "x": extra},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        """Return the cached response, or None on a miss or unreadable entry."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return LLMResponse(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def put(self, key: str, resp: LLMResponse) -> None:
        """Store a response (atomically, so concurrent writers never leave partial files)."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(resp), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def delete(self, key: str) -> None:
        """Remove an entry if present."""
        self._path(key).unlink(missing_ok=True)
