"""Pydantic models for the function calling pipeline."""

from typing import Any

from pydantic import BaseModel


class FunctionParam(BaseModel):
    """A single parameter or return type descriptor.

    Attributes:
        type: The JSON type name (e.g. ``"number"``, ``"string"``).
    """

    type: str


class FunctionDefinition(BaseModel):
    """Describes a callable function and its signature.

    Attributes:
        name: The function identifier.
        description: Human-readable purpose of the function.
        parameters: Mapping of parameter name to its descriptor.
        returns: Descriptor of the return value.
    """

    name: str
    description: str
    parameters: dict[str, FunctionParam]
    returns: FunctionParam


class Prompt(BaseModel):
    """A natural-language input prompt.

    Attributes:
        prompt: The raw user request string.
    """

    prompt: str


class FunctionCall(BaseModel):
    """A resolved function call with its arguments.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the function to invoke.
        parameters: Mapping of argument name to its value.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
