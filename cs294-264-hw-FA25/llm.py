from abc import ABC, abstractmethod
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


class LLM(ABC):
    """Abstract base class for Large Language Models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a prompt.
        Must include any required stop-token logic at the caller level.
        """
        raise NotImplementedError


class OpenAIModel(LLM):
    """Minimal OpenAI-backed LLM wrapper.

    Reads the API key from the environment (OPENAI_API_KEY). A project-level .env file
    can be used for local development.

    The model is expected to emit the function-call text that the ResponseParser
    can parse, including the end stop token.
    """

    def __init__(self, stop_token: str, model_name: str = "gpt-5-mini"):
        load_dotenv()  # best-effort: load from a project .env if present
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in environment or a .env file at project root."
            )
        self.client = OpenAI()
        self.stop_token = stop_token
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """Call the OpenAI Responses API (medium reasoning) and return plain text.

        The agent's system prompt instructs the model to output a single textual
        function call ending with the required stop token for the parser.
        """
        resp = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            reasoning={"effort": "medium"},
        )

        # Robust extraction of text across SDK versions
        try:
            if getattr(resp, "output_text", None):
                return resp.output_text  # type: ignore[attr-defined]
            parts: list[str] = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    txt = getattr(c, "text", None)
                    if txt:
                        parts.append(txt)
            return "\n".join(p for p in parts if p) or ""
        except Exception:
            return ""