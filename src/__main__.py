import json
import os
from sys import stderr, exit

from src.input import Input
from src.model import Model


def main() -> None:
    try:
        input_data = Input()
        model = Model(input_data)
    except Exception as e:
        print(f"Error during setup: {e}", file=stderr)
        exit(1)

    results: list[dict] = []

    for prompt in input_data.prompts:
        try:
            print(f"Processing prompt: {prompt.prompt}")
            function_call = model.choose_function_call(prompt.prompt)
            results.append({"prompt": prompt.prompt, **function_call})
            print(f"Result: {function_call}")
        except Exception as e:
            print(f"Error on prompt '{prompt.prompt}': {e}", file=stderr)
            continue

    output_path = input_data.args.output
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Output written to: {output_path}")
    except OSError as e:
        print(f"Error writing output: {e}", file=stderr)


if __name__ == "__main__":
    main()
