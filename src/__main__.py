import json
import os
from sys import stderr, exit

from src.input import Input
from src.model import Model


def main() -> None:
    """Runs the function-calling pipeline from CLI arguments to output file.

    Loads input data and the model, processes each prompt with constrained
    decoding, and writes the results as a JSON array. Errors on individual
    prompts are logged and skipped; setup errors exit with code 1.
    """
    try:
        input_data = Input.from_cli()
        model = Model(input_data)
    except Exception as e:
        print(f"Error during setup: {e}", file=stderr)
        exit(1)

    results: list[dict[str, object]] = []

    for prompt in input_data.prompts:
        try:
            func_call = model.choose_function_call(prompt.prompt)
            results.append({"prompt": prompt.prompt, **func_call.model_dump()})
        except Exception as e:
            print(f"Error on prompt '{prompt.prompt}': {e}", file=stderr)
            continue

    output_path = input_data.output_path
    try:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Output written to: {output_path}")
    except OSError as e:
        print(f"Error writing output: {e}", file=stderr)
        exit(1)


if __name__ == "__main__":
    main()
