import argparse
import json
from typing import Any
from pydantic import BaseModel, ConfigDict, ValidationError


class Prompt(BaseModel):
    prompt: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions_definition: list[FunctionDefinition]
    prompts: list[Prompt]
    output_path: str

    @classmethod
    def from_cli(cls) -> "Input":
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
        raw = cls._read_json_file(path)
        try:
            return [Prompt(**p) for p in raw]
        except (ValidationError, TypeError) as e:
            raise ValueError(f"Invalid prompt data in {path}: {e}")
