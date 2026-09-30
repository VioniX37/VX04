"""Routes each agent to the model configured for its role."""

from __future__ import annotations

from typing import Any

from automl_agent.config import ModelRole

from .base import LLMClient, LLMUsage


class LLMRouter:
    """Holds one client per role (`smart`, `fast`) that share a single usage counter.

    Agents declare a ``model_role``; the router hands them the matching client, so
    reasoning-heavy roles use the stronger model and high-volume roles the cheaper one.
    """

    def __init__(self, smart: LLMClient, fast: LLMClient | None = None) -> None:
        fast = fast or smart
        self.usage = LLMUsage()
        smart.usage = self.usage
        fast.usage = self.usage
        self._clients: dict[str, LLMClient] = {"smart": smart, "fast": fast}

    @property
    def provider(self) -> str:
        """Provider name of the underlying clients."""
        return self._clients["smart"].provider

    @property
    def model(self) -> str:
        """Human-readable summary of the models in use."""
        smart, fast = self._clients["smart"].model, self._clients["fast"].model
        return smart if smart == fast else f"{smart} / {fast}"

    def for_role(self, role: ModelRole) -> LLMClient:
        """Return the client for an agent role."""
        return self._clients[role]

    def models(self) -> dict[str, str]:
        """Model id per role."""
        return {role: client.model for role, client in self._clients.items()}

    def supports_search(self) -> bool:
        """Whether the smart client can do web-grounded search."""
        return hasattr(self._clients["smart"], "search")

    async def search(self, query: str) -> Any:
        """Delegate a grounded web search to the smart client."""
        return await self._clients["smart"].search(query)  # type: ignore[attr-defined]
