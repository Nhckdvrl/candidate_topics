from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .environment import ACTIONS

_ACTION_RE = re.compile(r"\b(A|B|C|WAIT)\b", re.IGNORECASE)


def parse_action(text: str) -> Optional[str]:
    stripped = text.strip().upper()
    if stripped in ACTIONS:
        return stripped
    m = _ACTION_RE.search(stripped)
    return m.group(1).upper() if m else None


@dataclass
class AgentReply:
    raw: str
    action: str
    valid: bool


class OpenAICompatibleAgent:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install openai>=1.40 to run the LLM experiment") from exc
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

    async def act(self, messages: List[Dict[str, str]]) -> AgentReply:
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content or ""
                action = parse_action(text)
                if action is None:
                    return AgentReply(raw=text, action="WAIT", valid=False)
                return AgentReply(raw=text, action=action, valid=True)
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                if attempt >= self.retries:
                    break
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")
