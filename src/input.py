import argparse
import json
from sys import stderr
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import field_validator


class Prompt(BaseModel):
    """A single natural language prompt to process.

    Attributes:
        prompt: The natural language request string.
    """

    prompt: str = Field(min_length=1, max_length=100)


class FunctionDefinition(BaseModel):
    """Describes a callable function with its parameters and return type.

    Attributes:
        name: The function identifier.
        description: A human-readable description of what the function does.
        parameters: A mapping of parameter names to their type metadata.
        returns: The return type metadata of the function.
    """

    name: str = Field(min_length=1, pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @field_validator('parameters')
    @classmethod
    def validate_parameter_types(
        cls, v: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        valid_types = {"string", "number", "boolean", "integer", "array",
                       "object"}
        for param_name, param_info in v.items():
            t = param_info.get('type', 'string')
            if t not in valid_types:
                raise ValueError(
                    f"Parameter '{param_name}' has invalid type '{t}', "
                    f"must be one of {valid_types}"
                )
        return v


class Input(BaseModel):
    """Holds all validated input data needed to run the program.

    Attributes:
        functions_definition: List of available functions the model can call.
        prompts: List of natural language prompts to process.
        output_path: Path where the output JSON file will be written.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions_definition: list[FunctionDefinition] = Field(min_length=1)
    prompts: list[Prompt] = Field(min_length=1)
    output_path: str

    @classmethod
    def from_cli(cls) -> "Input":
        """Creates an Input instance from CLI arguments and files.

        Returns:
            A validated Input instance populated from CLI arguments.

        Raises:
            ValueError: If any input file cannot be read or parsed.
        """
        args = cls._parse_args()
        functions_definition = cls._load_functions(args.functions_definition)
        prompts = cls._load_prompts(args.input)
        return cls(
            functions_definition=functions_definition,
            prompts=prompts,
            output_path=args.output,
        )

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        """Parses command-line arguments.

        Returns:
            A namespace with `functions_definition`, `input`, and `output`
            paths.
        """
        parser = argparse.ArgumentParser(description="Call Me Maybe")
        parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json",
            help="Path to the function definitions JSON file.",
        )
        parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json",
            help="Path to the input prompts JSON file.",
        )
        parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json",
            help="Path to the output JSON file.",
        )
        return parser.parse_args()

    @staticmethod
    def _read_json_file(file_path: str) -> Any:
        """Reads and parses a JSON file.

        Args:
            file_path: Path to the JSON file to read.

        Returns:
            The parsed JSON content.

        Raises:
            ValueError: If the file is not found, contains invalid JSON,
                or cannot be read.
        """
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except OSError as e:
            raise ValueError(f"Cannot read {file_path}: {e}")

    @classmethod
    def _load_functions(cls, path: str) -> list[FunctionDefinition]:
        """Loads and validates function definitions from a JSON file.

        Args:
            path: Path to the function definitions JSON file.

        Returns:
            A non-empty list of validated FunctionDefinition instances.

        Raises:
            ValueError: If the file is invalid, empty, or contains malformed
                function definitions.
        """
        raw = cls._read_json_file(path)
        try:
            result = [FunctionDefinition(**fn) for fn in raw]
        except (ValidationError, TypeError) as e:
            raise ValueError(f"Invalid function definition in {path}: {e}")
        if not result:
            raise ValueError(f"No function definitions found in {path}")
        return result

    @classmethod
    def _load_prompts(cls, path: str) -> list[Prompt]:
        """Loads and validates prompts from a JSON file.

        Args:
            path: Path to the prompts JSON file.

        Returns:
            A list of validated Prompt instances.

        Raises:
            ValueError: If the file is invalid or contains malformed prompts.
        """
        raw = cls._read_json_file(path)
        prompts = []
        for p in raw:
            try:
                prompts.append(Prompt(**p))
            except (ValidationError, TypeError) as e:
                print(f"Skipping invalid prompt: {e}", file=stderr)
        return prompts
