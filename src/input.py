import argparse
import json
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class Input:
    def __init__(self) -> None:
        self.parse_args()
        self.parse_functions_definition()
        self.parse_prompts()

    def parse_args(self) -> None:
        parser = argparse.ArgumentParser(
            description=("Call Me Maybe")
        )
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
            default="data/output/function_calls.json",
            help="Path to the output JSON file.",
        )
        self.args = parser.parse_args()

    @staticmethod
    def read_json_file(file_path: str) -> dict:
        with open(file_path, "r") as file:
            return json.load(file)

    def parse_functions_definition(self) -> None:
        raw_functions = self.read_json_file(self.args.functions_definition)
        self.functions_definition = []
        for func in raw_functions:
            self.functions_definition.append(FunctionDefinition(**func))

    def parse_prompts(self) -> None:
        raw_prompts = self.read_json_file(self.args.input)
        self.prompts = []
        for prompt in raw_prompts:
            self.prompts.append(Prompt(**prompt))
