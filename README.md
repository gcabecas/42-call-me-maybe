*This project has been created as part of the 42 curriculum by gcabecas.*

# call me maybe

> Introduction to function calling in LLMs

## Description

**call me maybe** is a function-calling system that translates natural language prompts into structured, typed function calls using a small language model (Qwen3-0.6B).

Instead of answering a question like *"What is the sum of 40 and 2?"* with *"42"*, the system outputs:

```json
{
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

The core challenge is reliability: small models (0.6B parameters) are unreliable at generating structured JSON on their own. This project solves that with **constrained decoding** — guiding the model token-by-token to guarantee 100% valid, schema-compliant output.

## Instructions

**Install dependencies:**
```bash
make install
```

**Run the program:**
```bash
make run
```

**Run with custom input/output paths:**
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

**Other Makefile targets:**

| Target | Description |
|--------|-------------|
| `make install` | Install dependencies via `uv sync` |
| `make run` | Run the main program |
| `make debug` | Run with Python's `pdb` debugger |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |
| `make lint` | Run `flake8` + `mypy` with standard flags |
| `make lint-strict` | Run `flake8` + `mypy --strict` |

Input files must be placed in `data/input/`. Output is written to `data/output/function_calling_results.json`.

## Algorithm Explanation

### Constrained Decoding

At each generation step, the model produces a probability distribution (logits) over all possible next tokens. Constrained decoding intervenes **before** token selection by masking out invalid tokens (setting their logits to −∞), so only structurally valid tokens can be chosen.

The pipeline follows:
```
Prompt → Tokenization → Input IDs → LLM → Logits → Constrained selection → Next token
```

This loop repeats until the full output is generated.

### Function Name Selection

Rather than generating the function name token-by-token freely, the system computes a **log-probability score** for each candidate function by evaluating the model's logits at each token position of the name. The function with the highest cumulative score is selected. This avoids any risk of the model generating a non-existent function name.

A **confidence threshold** of `log(0.8)` is applied to the normalized score (average log-probability per token). If the best-scoring function falls below this threshold, the prompt is considered unmatched, a `ValueError` is raised, and the prompt is skipped. This was calibrated empirically: valid prompts score between -0.065 and ~0, while unrelated prompts score below -0.389.

### Parameter Decoding

Each parameter is decoded according to its declared type:

- **Numbers**: only tokens that extend a valid numeric prefix are allowed (e.g. digits, `.`, `e`, `+`, `-`). Decoding stops when the model naturally selects a terminator (`,`, `}`, space, etc.).
- **Strings**: free decoding until an unescaped `"` or newline is reached.
- **Booleans**: constrained to tokens that form a prefix of `"true"` or `"false"`.

If the token limit is reached without a valid termination, a `ValueError` is raised and the prompt is skipped.

## Design Decisions

- **No private SDK methods**: only the public API of `llm_sdk` is used (`get_logits_from_input_ids`, `encode`, `decode`, `get_path_to_vocabulary_json`).
- **Pydantic validation**: all classes use Pydantic — `FunctionCall` and `Input`-related classes are `BaseModel`, and `Model.__init__` uses `@validate_call` for input validation.
- **Error isolation**: decoding errors raise `ValueError`, caught in `__main__` so one failing prompt never crashes the whole run.
- **NumPy for logit operations**: vectorized masking and argmax over the vocabulary for efficiency.
- **ChatML prompt format**: the prompt is structured using `<|im_start|>` / `<|im_end|>` tokens, which match the format Qwen3 was trained on.

## Performance Analysis

| Metric | Target |
|--------|--------|
| Function selection accuracy | 90%+ |
| Output JSON validity | 100% |
| Processing time (all prompts) | < 5 minutes |

Constrained decoding guarantees structurally valid JSON regardless of model confidence, separating correctness of structure from correctness of semantics.

## Challenges Faced

**Prompt engineering**: early prompt designs were too complex, leading the model to generate confused or verbose outputs before reaching the function name. The solution was a simplified ChatML-format prompt that directly prefixes `fn_name:` in the assistant turn, steering the model immediately toward structured output.

**Simplifying Qwen interaction**: the first version of the code involved more complex interactions with the model. A second version was refactored to reduce this complexity, making the constrained decoding pipeline cleaner and easier to reason about.

## Testing Strategy

- **Functional testing**: 11 prompts in `data/input/function_calling_tests.json` covering all 5 available functions, various parameter types (numbers, strings, booleans), and edge cases.
- **Output validation**: verify that `data/output/function_calling_results.json` is valid JSON, contains all expected keys (`prompt`, `name`, `parameters`), and that types match the function definitions.
- **Static analysis**: `make lint` and `make lint-strict` ensure type correctness via `mypy` and code style via `flake8`.

## Example Usage

```bash
$ make run
Processing prompt: What is the sum of 2 and 3?
Result: name='fn_add_numbers' parameters={'a': 2.0, 'b': 3.0}
Processing prompt: Greet shrek
Result: name='fn_greet' parameters={'name': 'shrek'}
Processing prompt: Reverse the string 'hello'
Result: name='fn_reverse_string' parameters={'s': 'hello'}
Output written to: data/output/function_calling_results.json
```

Output file (`data/output/function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  }
]
```

## Resources

### References
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [Qwen3](https://qwen.readthedocs.io/en/v2.0/getting_started/concepts.html)
- [Outlines — structured generation library](https://github.com/dottxt-ai/outlines) *(reference only — not used in this project)*

### AI Usage
AI was used in this project for:
- **Understanding concepts**: clarifying how constrained decoding works, how tokenizers handle subword units.
- **Documentation**: assistance in writing and structuring this README.

All generated content was reviewed, tested, and fully understood before being included in the project.
