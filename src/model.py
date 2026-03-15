import json

import numpy as np

from llm_sdk import Small_LLM_Model
from .input import Input


class Model():
    def __init__(self, input_data: Input):
        self.input_data = input_data
        self.model = Small_LLM_Model()

        vocab_path = self.model.get_path_to_vocab_file()

        try:
            with open(vocab_path, "r") as f:
                raw_vocab: dict[str, int] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load vocabulary: {e}")

        number_ids: list[int] = []
        bool_ids: list[int] = []
        rev_vocab: dict[int, str] = {}

        for tok_str, tok_id in raw_vocab.items():
            useful = False

            if self._is_number_token(tok_str):
                number_ids.append(tok_id)
                useful = True

            if any(lit.startswith(tok_str) for lit in ("true", "false")):
                bool_ids.append(tok_id)
                useful = True

            if useful:
                rev_vocab[tok_id] = tok_str

        self.rev_vocab = rev_vocab
        self.number_ids = np.array(number_ids, dtype=np.int32)
        self.bool_ids = bool_ids
        self.functions_desc = self._format_functions()

    @staticmethod
    def _is_number_token(tok: str) -> bool:
        return bool(tok) and all(c in "0123456789.eE+-" for c in tok)

    @staticmethod
    def _is_number_prefix(s: str) -> bool:
        if not s or s == '-':
            return True
        if s[0] not in '-0123456789':
            return False
        padded = s + '0' if s[-1] in '.eE+-' else s
        try:
            float(padded)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_complete_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def _unescaped_quote_idx(s: str) -> int | None:
        i = 0
        while i < len(s):
            if s[i] == '\\':
                i += 2
                continue
            if s[i] == '"':
                return i
            i += 1
        return None

    def _format_functions(self) -> str:
        lines = []
        for fn in self.input_data.functions_definition:
            params = ", ".join(
                f"{k}: {v['type']}" for k, v in fn.parameters.items()
            )
            lines.append(f"- {fn.name}({params}): {fn.description}")
        return "\n".join(lines)

    def _build_params_prompt_ids(
            self,
            user_prompt: str,
            partial_json: str) -> list[int]:
        full_prompt = (
            "You are a function-calling assistant. "
            "Extract the arguments from the user request.\n\n"
            "Available functions:\n"
            f"{self.functions_desc}\n\n"
            "Rules:\n"
            "- Output ONLY the JSON object. "
            "No explanation, no extra text.\n"
            "- string values must be quoted.\n"
            "- regex values must be valid Python regex patterns. "
            "Use character classes, not literal values "
            "(e.g. \"\\\\d+\" for digits, "
            "\"[aeiouAEIOU]\" for vowels, "
            "\"\\\\w+\" for words).\n\n"
            "Examples:\n"
            "User request: Say hello to Alice\n"
            "JSON: {\"name\": \"fn_greet\", "
            "\"parameters\": {\"name\": \"Alice\"}}\n\n"
            "User request: Replace all numbers in 'foo 42 bar' with X\n"
            "JSON: {\"name\": \"fn_substitute_string_with_regex\", "
            "\"parameters\": {\"source_string\": \"foo 42 bar\", "
            "\"regex\": \"\\\\d+\", \"replacement\": \"X\"}}\n\n"
            "User request: Replace vowels in 'Hello' with *\n"
            "JSON: {\"name\": \"fn_substitute_string_with_regex\", "
            "\"parameters\": {\"source_string\": \"Hello\", "
            "\"regex\": \"[aeiouAEIOU]\", "
            "\"replacement\": \"*\"}}\n\n"
            f"User request: {user_prompt}\n"
            f"JSON: {partial_json}"
        )
        return self.model.encode(full_prompt)[0].tolist()

    def _decode_fn_name(self, user_prompt: str) -> str:
        fn_names = [
            fn.name for fn in self.input_data.functions_definition
        ]
        base_ids = self._build_params_prompt_ids(
            user_prompt, '{"name": "'
        )

        # Compute base logits once — shared first-token position for all
        base_logits = self.model.get_logits_from_input_ids(base_ids)

        best_name = fn_names[0]
        best_score = float('-inf')

        for name in fn_names:
            name_ids = self.model.encode(name)[0].tolist()
            score = float(base_logits[name_ids[0]])
            current_ids = base_ids + [name_ids[0]]

            for tok_id in name_ids[1:]:
                logits = self.model.get_logits_from_input_ids(current_ids)
                score += logits[tok_id]
                current_ids.append(tok_id)

            score /= max(len(name_ids), 1)

            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def _decode_number(
            self, input_ids: list[int]) -> tuple[float, list[int]]:
        generated = ""
        current_ids = list(input_ids)
        terminators = set(',} \n\t"')

        for _ in range(15):
            logits = self.model.get_logits_from_input_ids(current_ids)
            logits_arr = np.array(logits)

            unconstrained_best = int(np.argmax(logits_arr))
            unconstrained_str = self.model.decode([unconstrained_best])
            if (self._is_complete_number(generated)
                    and unconstrained_str
                    and unconstrained_str[0] in terminators):
                break

            valid_ids = self.number_ids[
                [self._is_number_prefix(
                    generated + self.rev_vocab[int(tid)]
                ) for tid in self.number_ids]
            ]
            if len(valid_ids) == 0:
                break

            best_id = int(valid_ids[np.argmax(logits_arr[valid_ids])])
            generated += self.rev_vocab[best_id]
            current_ids.append(best_id)

        try:
            return float(generated), current_ids
        except ValueError:
            return 0.0, current_ids

    def _decode_string(
            self, input_ids: list[int]) -> tuple[str, list[int]]:
        generated = ""
        current_ids = list(input_ids)

        for _ in range(100):
            logits = self.model.get_logits_from_input_ids(current_ids)
            best_id = int(np.argmax(logits))
            best_str = self.model.decode([best_id])

            if '\n' in best_str:
                break

            quote_idx = self._unescaped_quote_idx(best_str)
            if quote_idx is not None:
                generated += best_str[:quote_idx]
                current_ids.append(best_id)
                break

            generated += best_str
            current_ids.append(best_id)

        return generated, current_ids

    def _decode_boolean(
            self, input_ids: list[int]) -> tuple[bool, list[int]]:
        generated = ""
        current_ids = list(input_ids)
        literals = ("true", "false")

        for _ in range(10):
            valid_ids = [
                tid for tid in self.bool_ids
                if any(
                    lit.startswith(generated + self.rev_vocab[tid])
                    for lit in literals
                )
            ]
            if not valid_ids:
                break

            logits = self.model.get_logits_from_input_ids(current_ids)
            best_id = max(valid_ids, key=lambda tid: logits[tid])
            generated += self.rev_vocab[best_id]
            current_ids.append(best_id)

            if generated in literals:
                break

        return generated == "true", current_ids

    def choose_function_call(self, prompt: str) -> dict:
        fn_name = self._decode_fn_name(prompt)

        fn_def = next(
            fn for fn in self.input_data.functions_definition
            if fn.name == fn_name
        )

        prefix = f'{{"name": "{fn_name}", "parameters": {{'
        input_ids = self._build_params_prompt_ids(prompt, prefix)

        parameters: dict = {}
        for i, (param_name, param_info) in enumerate(
            fn_def.parameters.items()
        ):
            sep = ", " if i > 0 else ""
            param_type = param_info.get("type", "string")

            if param_type == "string":
                input_ids = self._extend_ids(
                    input_ids, f'{sep}"{param_name}": "'
                )
                value, input_ids = self._decode_string(input_ids)
            elif param_type == "number":
                input_ids = self._extend_ids(
                    input_ids, f'{sep}"{param_name}": '
                )
                value, input_ids = self._decode_number(input_ids)
            elif param_type == "boolean":
                input_ids = self._extend_ids(
                    input_ids, f'{sep}"{param_name}": '
                )
                value, input_ids = self._decode_boolean(input_ids)
            else:
                input_ids = self._extend_ids(
                    input_ids, f'{sep}"{param_name}": "'
                )
                value, input_ids = self._decode_string(input_ids)

            parameters[param_name] = value

        return {"name": fn_name, "parameters": parameters}

    def _extend_ids(
            self, input_ids: list[int], text: str) -> list[int]:
        return input_ids + self.model.encode(text)[0].tolist()
