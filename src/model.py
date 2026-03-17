import json
from typing import Any
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

        self._fn_sigs_str: str = ", ".join(
            "{}({})".format(
                fn.name,
                ", ".join(
                    f"{k}: {v['type']}" for k, v in fn.parameters.items()
                )
            )
            for fn in self.input_data.functions_definition
        )
        self._fn_names: list[str] = [
            fn.name for fn in self.input_data.functions_definition
        ]
        _tids = [
            self.model.encode(name)[0].tolist() for name in self._fn_names
        ]
        common_len = 0
        while (common_len < min(len(t) for t in _tids)
               and len({t[common_len] for t in _tids}) == 1):
            common_len += 1
        self._fn_common_prefix_ids: list[int] = _tids[0][:common_len]
        self._fn_tids_short: list[list[int]] = [t[common_len:] for t in _tids]
        self._fn_name_groups: list[dict[tuple[int, ...], list[int]]] = []
        for depth in range(max(len(t) for t in self._fn_tids_short)):
            groups: dict[tuple[int, ...], list[int]] = {}
            for i, t in enumerate(self._fn_tids_short):
                if depth < len(t):
                    groups.setdefault(tuple(t[:depth]), []).append(i)
            self._fn_name_groups.append(groups)

        self._fn_lookup = {
            fn.name: fn for fn in self.input_data.functions_definition
        }
        self._fn_ids: dict[str, tuple[list[int], list[list[int]]]] = {}
        for fn in self.input_data.functions_definition:
            hint = (
                "\nregex: [bracket] notation, replacement: exact symbol"
                if fn.name == "fn_substitute_string_with_regex" else ""
            )
            preamble = (
                f"fn_name: {fn.name}{hint}\n"
                f'{{\"name\": \"{fn.name}\", \"parameters\": {{'
            )
            param_prefix_ids: list[list[int]] = []
            for i, (param_name, param_info) in enumerate(
                    fn.parameters.items()):
                sep = ", " if i > 0 else ""
                suffix = '"' if param_info.get(
                    "type", "string") == "string" else ""
                param_prefix_ids.append(self.model.encode(
                    f'{sep}"{param_name}": {suffix}')[0].tolist())
            self._fn_ids[fn.name] = (
                self.model.encode(preamble)[0].tolist(),
                param_prefix_ids,
            )

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

    def _build_prompt_ids(self, user_prompt: str) -> list[int]:
        prompt = (
            f"<|im_start|>find the correct function call and re arguments\n"
            f"{self._fn_sigs_str}\n"
            f"{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<tool_call>\n\n</tool_call>\n"
            f"fn_name: ")
        return self.model.encode(prompt)[0].tolist()

    def _decode_fn_name(self, base_ids: list[int]) -> str:
        scoring_base = base_ids + self._fn_common_prefix_ids
        scores = [0.0] * len(self._fn_names)
        for depth, groups in enumerate(self._fn_name_groups):
            for prefix, idxs in groups.items():
                logits_arr = np.array(
                    self.model.get_logits_from_input_ids(
                        scoring_base + list(prefix)))
                log_probs = logits_arr - (
                    logits_arr.max()
                    + np.log(np.sum(np.exp(logits_arr - logits_arr.max())))
                )
                for i in idxs:
                    tid = self._fn_tids_short[i][depth]
                    scores[i] += float(log_probs[tid])
        return self._fn_names[
            max(range(len(self._fn_names)), key=lambda i: scores[i])
        ]

    def _decode_number(self, input_ids: list[int]) -> tuple[float, list[int]]:
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

    def _decode_string(self, input_ids: list[int]) -> tuple[str, list[int]]:
        generated = ""
        current_ids = list(input_ids)
        for _ in range(20):
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
        return generated.strip(), current_ids

    def _decode_boolean(self, input_ids: list[int]) -> tuple[bool, list[int]]:
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

    def choose_function_call(self, prompt: str) -> dict[str, Any]:
        base_ids = self._build_prompt_ids(prompt)
        fn_name = self._decode_fn_name(base_ids)
        fn_def = self._fn_lookup[fn_name]
        preamble_ids, param_prefix_ids = self._fn_ids[fn_name]
        input_ids = base_ids + preamble_ids

        parameters: dict[str, str | float | bool] = {}
        for (param_name, param_info), prefix_ids in zip(
                fn_def.parameters.items(), param_prefix_ids):
            input_ids = input_ids + prefix_ids
            param_type = param_info.get("type", "string")
            value: str | float | bool
            if param_type == "number":
                value, input_ids = self._decode_number(input_ids)
            elif param_type == "boolean":
                value, input_ids = self._decode_boolean(input_ids)
            else:
                value, input_ids = self._decode_string(input_ids)
            parameters[param_name] = value

        return {"name": fn_name, "parameters": parameters}
