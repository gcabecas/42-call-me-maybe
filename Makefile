.PHONY: run install debug clean lint lint-strict

INPUT     ?= data/input/function_calling_tests.json
OUTPUT    ?= data/output/function_calling_results.json
FUNCTIONS ?= data/input/functions_definition.json

run:
	uv run python -m src --input $(INPUT) --output $(OUTPUT) --functions_definition $(FUNCTIONS)

install:
	uv sync

debug:
	uv run python -m pdb -m src --input $(INPUT) --output $(OUTPUT) --functions_definition $(FUNCTIONS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

lint:
	uv run flake8 src/
	uv run mypy src/ \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict
