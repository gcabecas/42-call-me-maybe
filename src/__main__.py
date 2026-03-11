"""Entry point for the call-me-maybe function calling tool."""

import argparse

from .file_io import load_function_definitions, load_prompts, save_results
from .models import FunctionCall


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with ``functions_definition``, ``input``,
        and ``output`` path attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompts into function calls."
        )
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
    return parser.parse_args()


def main() -> None:
    """Run the function calling pipeline."""
    args = parse_args()

    definitions = load_function_definitions(args.functions_definition)
    prompts = load_prompts(args.input)

    print(f"Loaded {len(definitions)} function definitions, definitions:"
          f" {[d.name for d in definitions]}")

    print(f"Loaded {len(prompts)} prompts, prompts: "
          f"{[p.prompt for p in prompts]}")

    results: list[FunctionCall] = []

    save_results(args.output, results)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
