"""Settings for the database and the two language models, read from ``.env``.

``.env.example`` lists every variable and explains which model settings go
together. Each settings class reads the environment when it is created.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ROOT_PATH = Path(__file__).resolve().parent.parent
# override=True: values in .env win over variables already set in the shell.
load_dotenv(_ROOT_PATH / ".env", override=True)

# Allowed values of OPENAI_REASONING_EFFORT / JUDGE_REASONING_EFFORT. Checked at
# startup so a typo fails immediately, not as an API error halfway through a run.
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def _optional(name: str) -> str | None:
    """Read an environment variable, treating an empty value as "not set".

    This matters for model settings: sending an empty temperature to the API is an error, so a blank line in ``.env``
    must mean "don't send it".

    Args:
        name: Variable name.

    Returns:
        str | None: The value without surrounding spaces, or None if the
        variable is missing or blank.
    """
    return os.getenv(name, "").strip() or None


def _optional_float(name: str) -> float | None:
    """Read an optional numeric environment variable, e.g. a temperature.

    Args:
        name: Variable name.

    Returns:
        float | None: The number, or None if the variable is missing or blank.

    Raises:
        ValueError: The variable is set but is not a number.
    """
    value = _optional(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name}={value!r} is not a number; set a value like 0 or leave it blank.")


def _reasoning_effort(name: str) -> str | None:
    """Read a reasoning-effort environment variable and check its value.

    Args:
        name: Variable name.

    Returns:
        str | None: One of ``REASONING_EFFORTS``, or None if the variable is
        missing or blank.

    Raises:
        ValueError: The value is not one of ``REASONING_EFFORTS``.
    """
    value = _optional(name)
    if value is not None and value not in REASONING_EFFORTS:
        raise ValueError(
            f"{name}={value!r} is not a valid reasoning effort; "
            f"use one of {', '.join(REASONING_EFFORTS)} or leave it blank."
        )
    return value


@dataclass
class DatabaseSettings:
    """Where and as whom to connect to PostgreSQL (DB_* variables in ``.env``)."""

    dbname: str = field(default_factory=lambda: os.getenv("DB_NAME", ""))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", 5432)))

    def to_dict(self) -> dict[str, str | int]:
        """Return the settings in the form the psycopg connection pool expects.

        Returns:
            dict[str, str | int]: ``dbname``, ``user``, ``password``, ``host``
            and ``port``, ready to pass as connection arguments.
        """
        return {
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
        }


@dataclass
class OpenAISettings:
    """What the settings of every OpenAI model share: the one API key (``OPENAI_API_KEY``).

    Only the key lives here. Model, temperature and reasoning effort stay in each subclass, because each reads its own
    variables and has its own defaults.
    """

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    def require_api_key(self) -> str:
        """Return the OpenAI API key, or fail with a clear message if it is missing.

        Without this check a missing key would only surface later, as a
        harder-to-read authentication error from the OpenAI library.

        Returns:
            str: The OpenAI API key.

        Raises:
            RuntimeError: OPENAI_API_KEY is missing or empty.
        """
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — copy it into .env (see .env.example).")
        return self.api_key


@dataclass
class AgentSettings(OpenAISettings):
    """Settings of the model that answers questions (OPENAI_* variables in ``.env``).

    A model accepts either a temperature or a reasoning effort, never both; ``.env.example`` explains which. If the
    reasoning effort is left blank it becomes ``"none"``, because OpenAI allows tool calling on reasoning models only at
    that level — and the agent cannot work without tools.
    """

    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    temperature: float | None = field(default_factory=lambda: _optional_float("OPENAI_TEMPERATURE"))
    reasoning_effort: str | None = field(default_factory=lambda: _reasoning_effort("OPENAI_REASONING_EFFORT") or "none")


@dataclass
class EmbeddingSettings(OpenAISettings):
    """Settings of the model that turns course card sections and search queries into vectors (``EMBEDDING_MODEL``).

    Stored vectors are only comparable with vectors of the same model, so changing the model means recomputing every
    stored one — the next ingestion run does that by itself. The search needs the key (``require_api_key``), but
    ingestion runs without it: the embedding step is then skipped.
    """

    model: str = field(default_factory=lambda: _optional("EMBEDDING_MODEL") or "text-embedding-3-large")


@dataclass
class JudgeSettings(OpenAISettings):
    """Settings of the model that scores answers during evaluation (JUDGE_* in ``.env``).

    Kept separate from ``AgentSettings`` so the judge can be a different model. Unlike the agent, a blank reasoning
    effort stays blank (the model's own default is used): the judge does not call tools, so it has no such limit.
    """

    model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-6-luna"))
    temperature: float | None = field(default_factory=lambda: _optional_float("JUDGE_TEMPERATURE"))
    reasoning_effort: str | None = field(default_factory=lambda: _reasoning_effort("JUDGE_REASONING_EFFORT"))
