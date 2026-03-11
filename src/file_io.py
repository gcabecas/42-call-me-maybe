"""File I/O utilities for loading and saving JSON data."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import FunctionCall, FunctionDefinition, Prompt


def load_function_definitions(
    path: str,
) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Args:
        path: Path to the JSON file containing function definitions.

    Returns:
        List of validated FunctionDefinition instances.

    Raises:
        SystemExit: If the file is missing, unreadable, contains
            invalid JSON, or does not match the expected schema.
    """
    raw = _read_json(path)
    try:
        return [FunctionDefinition.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise SystemExit(f"Schema error in {path}: {exc}") from exc


def load_prompts(path: str) -> list[Prompt]:
    """Load natural-language prompts from a JSON file.

    Args:
        path: Path to the JSON file containing prompt objects.

    Returns:
        List of validated Prompt instances.

    Raises:
        SystemExit: If the file is missing, unreadable, contains
            invalid JSON, or does not match the expected schema.
    """
    raw = _read_json(path)
    try:
        return [Prompt.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise SystemExit(f"Schema error in {path}: {exc}") from exc


def save_results(path: str, results: list[FunctionCall]) -> None:
    """Write function call results to a JSON file.

    Args:
        path: Destination file path for the output JSON.
        results: List of FunctionCall instances to serialise.

    Raises:
        SystemExit: If the directory cannot be created or the file
            cannot be written.
    """
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in results]
        output_path.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        raise SystemExit(f"Could not write to {path}: {exc}") from exc


def _read_json(path: str) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialised Python object.

    Raises:
        SystemExit: If the file is missing or contains invalid JSON.
    """
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
