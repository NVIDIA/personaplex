# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from typing import AsyncGenerator, List, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_KOREAN_SYSTEM_PROMPT = (
    "당신은 친절한 AI 비서입니다. 사용자의 질문에 한국어로 자연스럽게 대답하세요. "
    "답변은 간결하고 대화체로 해주세요."
)


class KoreanLLM:
    """Korean LLM using OpenAI-compatible API (works with vLLM, Ollama, or cloud)."""

    def __init__(
        self,
        endpoint: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        api_key: str = "ollama",
        max_tokens: int = 256,
    ):
        import openai

        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
        self._conversation_history: List[Dict[str, str]] = []
        logger.info(
            f"KoreanLLM initialized: endpoint={endpoint}, model={model}"
        )

    def reset(self):
        self._conversation_history = []

    async def generate(
        self,
        user_text: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a Korean response via streaming.

        Args:
            user_text: User's message in Korean.
            system_prompt: Optional system prompt override. Defaults to Korean assistant prompt.

        Yields:
            Text chunks as they arrive from the LLM.
        """
        if system_prompt is None:
            system_prompt = DEFAULT_KOREAN_SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": user_text})

        self._conversation_history.append({"role": "user", "content": user_text})

        full_response = []
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True,
            temperature=0.7,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text_piece = chunk.choices[0].delta.content
                full_response.append(text_piece)
                yield text_piece

        assistant_message = "".join(full_response)
        self._conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        # Keep conversation history manageable
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-16:]
