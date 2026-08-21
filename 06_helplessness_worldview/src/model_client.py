from __future__ import annotations

import asyncio
import json
import random
import re
import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import httpx


class ChatClient(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class OpenAICompatibleClient:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.7
    max_tokens: int = 40
    timeout: float = 120.0
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        root = self.base_url.rstrip("/")
        url = root + "/chat/completions" if root.endswith("/v1") else root + "/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        r = await self._client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@dataclass
class MockClient:
    """Pipeline smoke-test client. It is not a scientific baseline."""
    seed: int = 0

    async def complete(self, messages: list[dict[str, str]]) -> str:
        await asyncio.sleep(0)
        last = messages[-1]["content"]
        actions = extract_available_actions(last)
        if not actions:
            return '{"action":"invalid"}'
        raw = f"{self.seed}|{len(messages)}|{last}".encode("utf-8")
        deterministic = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        rng = random.Random(deterministic)
        return json.dumps({"action": rng.choice(actions)})


def extract_available_actions(text: str) -> list[str]:
    m = re.search(r"Choose one action:\s*([^,]+),\s*([^,]+),\s*or\s*([^\.]+)", text)
    if not m:
        return []
    return [m.group(i).strip() for i in (1, 2, 3)]


def parse_surface_action(raw: str, allowed: tuple[str, str, str]) -> tuple[str, bool]:
    """Return (action, valid). Invalid responses use deterministic active fallback A."""
    try:
        obj = json.loads(raw)
        action = str(obj.get("action", ""))
        if action in allowed:
            return action, True
    except Exception:
        pass
    lower = raw.lower()
    hits = [a for a in allowed if a.lower() in lower]
    if len(hits) == 1:
        return hits[0], True
    return allowed[0], False
