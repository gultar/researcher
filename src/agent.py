from openai import OpenAI
from dotenv import load_dotenv
from datetime import date
from typing import Optional
import os
import yaml

load_dotenv()


def _load_llm_params():
    """Read LLM provider settings from parameters.yaml."""
    try:
        with open("./parameters.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        return params
    except Exception:
        return {}


def create_client(params: dict = None):
    """
    Create an OpenAI-compatible client based on the llm_provider setting.

    Supported providers:
      - "openai"  : uses the standard OpenAI API (requires OPENAI_API_KEY env var)
      - "ollama"  : uses a local Ollama server via its OpenAI-compatible endpoint
                    (set ollama_base_url in parameters.yaml, default: http://localhost:11434)
    """
    if params is None:
        params = _load_llm_params()

    provider = params.get("llm_provider", "openai").lower()

    if provider == "ollama":
        base_url = params.get("ollama_base_url", "http://localhost:11434")
        # Ollama exposes an OpenAI-compatible endpoint at /v1
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        return OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama does not require a real key
        )
    else:
        # Standard OpenAI - key is read from OPENAI_API_KEY env var
        return OpenAI()


class Agent:
    def __init__(
        self,
        role="general",
        model=None,
        smaller_model=None,
        params: dict = None,
    ):
        if params is None:
            params = _load_llm_params()

        self.params = params
        self.initial_instructions = "You are a helpful assistant."
        self.messages = [{"role": "system", "content": self.initial_instructions}]
        self.role = role

        # Model names: prefer explicit args, fall back to parameters.yaml, then hard defaults
        provider = params.get("llm_provider", "openai").lower()
        if model is None:
            if provider == "ollama":
                model = params.get("llm_model", "llama3")
            else:
                model = params.get("llm_model", "gpt-4o")
        if smaller_model is None:
            if provider == "ollama":
                smaller_model = params.get("llm_smaller_model", model)
            else:
                smaller_model = params.get("llm_smaller_model", "gpt-4o-mini")

        self.model = model
        self.smaller_model = smaller_model
        self.client = create_client(params)

        today = date.today()
        today = str(today)

    def log(self, text):
        return print(text)

    def record(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        self.log(prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def complete(self, prompt: str, temperature: Optional[float] = 0):
        messages = [
            {"role": "system", "content": self.initial_instructions},
            {"role": "user", "content": prompt},
        ]
        self.log(prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def cheaper_complete(self, prompt: str, temperature: Optional[float] = 0):
        messages = [
            {"role": "system", "content": self.initial_instructions},
            {"role": "user", "content": prompt},
        ]
        self.log(prompt)
        response = self.client.chat.completions.create(
            model=self.smaller_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
