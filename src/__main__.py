from sys import stderr, exit
from src.input import Input


def main() -> None:
    try:
        input_data = Input()
        print("Input data loaded successfully.")
    except Exception as e:
        print(f"Error loading input data: {e}", file=stderr)
        exit(1)

    print("Function definitions:")
    print(input_data.functions_definition)
    print("\nPrompts:")
    print(input_data.prompts)


if __name__ == "__main__":
    main()
