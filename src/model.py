import json
import numpy as np
from pydantic import BaseModel, validate_call

from llm_sdk import Small_LLM_Model
from .input import Input


class FunctionCall(BaseModel):
    """Represents a structured function call produced by the model.

    Attributes:
        name: The name of the function to call.
        parameters: A mapping of parameter names to their typed values.
    """

    name: str
    parameters: dict[str, str | float | int | bool]


class Model():
    """Loads the LLM and vocabulary, and performs constrained decoding.

    Attributes:
        input_data: The validated input containing function definitions and
        prompts.
    """

    _MAX_STEPS: dict[str, int] = {"number": 15, "boolean": 10, "string": 20}

    @validate_call
    def __init__(self, input_data: Input, show: bool = False) -> None:
        self.input_data = input_data
        self._show = show
        self._llm = Small_LLM_Model()
        self._load_vocab()
        self._build_fn_name_tokens()
        self._build_fn_param_tokens()

    def _load_vocab(self) -> None:
        """Loads the vocabulary and categorizes tokens for numeric, boolean,
        and reverse-lookup use."""
        vocab_path = self._llm.get_path_to_vocab_file()
        try:
            with open(vocab_path, "r") as f:
                raw_vocab: dict[str, int] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load vocabulary: {e}")

        number_ids: list[int] = []
        bool_ids: list[int] = []
        rev_vocab: dict[int, str] = {}
        for tok_str, tok_id in raw_vocab.items():
            is_num = self._is_number_token(tok_str)
            is_bool = any(lit.startswith(tok_str) for lit in ("true", "false"))
            if is_num:
                number_ids.append(tok_id)
            if is_bool:
                bool_ids.append(tok_id)
            if is_num or is_bool:
                rev_vocab[tok_id] = tok_str

        self._rev_vocab = rev_vocab
        self._number_ids = np.array(number_ids, dtype=np.int32)
        self._bool_ids = bool_ids

    def _build_fn_name_tokens(self) -> None:
        """Precomputes function signatures, names, and token prefix data for
        name decoding."""
        self._fn_sigs_str = ", ".join(
            "{}({})".format(
                fn.name,
                ", ".join(
                    f"{k}: {v['type']}" for k, v in fn.parameters.items()
                )
            )
            for fn in self.input_data.functions_definition
        )
        self._fn_names = [
            fn.name for fn in self.input_data.functions_definition]
        _tids = [self._llm.encode(name)[0].tolist() for name in self._fn_names]
        common_len = 0
        while (common_len < min(len(t) for t in _tids)
               and len({t[common_len] for t in _tids}) == 1):
            common_len += 1
        self._fn_common_prefix_ids = _tids[0][:common_len]
        self._fn_tids_short = [t[common_len:] for t in _tids]
        self._fn_name_groups = []
        for depth in range(max(len(t) for t in self._fn_tids_short)):
            groups: dict[tuple[int, ...], list[int]] = {}
            for i, t in enumerate(self._fn_tids_short):
                if depth < len(t):
                    groups.setdefault(tuple(t[:depth]), []).append(i)
            self._fn_name_groups.append(groups)

    def _build_fn_param_tokens(self) -> None:
        """Precomputes preamble and parameter prefix token IDs for each
        function."""
        self._fn_lookup = {
            fn.name: fn for fn in self.input_data.functions_definition
        }
        self._fn_ids = {}
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
                param_prefix_ids.append(
                    self._llm.encode(
                        f'{sep}"{param_name}": {suffix}'
                    )[0].tolist())
            self._fn_ids[fn.name] = (
                self._llm.encode(preamble)[0].tolist(),
                param_prefix_ids,
            )

    @staticmethod
    def _is_number_token(tok: str) -> bool:
        """Checks whether a token string consists only of numeric characters.

        Args:
            tok: The token string to check.

        Returns:
            True if the token is non-empty and contains only digits or numeric
            symbols (`.`, `e`, `E`, `+`, `-`).
        """
        return bool(tok) and all(c in "0123456789.eE+-" for c in tok)

    @staticmethod
    def _is_number_prefix(s: str) -> bool:
        """Checks whether a string is a valid prefix of a float literal.

        Args:
            s: The string to check.

        Returns:
            True if the string could be the beginning of a valid float number.
        """
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
        """Checks whether a string is a complete, parseable float.

        Args:
            s: The string to check.

        Returns:
            True if the string can be parsed as a float.
        """
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def _unescaped_quote_idx(s: str) -> int | None:
        """Finds the index of the first unescaped double-quote in a string.

        Args:
            s: The string to search.

        Returns:
            The index of the first unescaped `"`, or None if not found.
        """
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
        """Builds the token ID sequence for the full prompt.

        Formats the prompt in ChatML style with function signatures and
        the user request, then encodes it into token IDs.

        Args:
            user_prompt: The natural language prompt from the user.

        Returns:
            A list of token IDs representing the full prompt.
        """
        prompt = (
            f"<|im_start|>system find the correct function call and re"
            f"arguments\n"
            f"{self._fn_sigs_str}\n"
            f"{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<tool_call>\n\n</tool_call>\n"
            f"fn_name: ")
        return self._llm.encode(prompt)[0].tolist()

    def _decode_fn_name(self, base_ids: list[int]) -> str:
        """Selects the most likely function name using log-probability scoring.

        Scores each candidate function name by accumulating log-probabilities
        at each token position, then returns the name with the highest score,
        provided it exceeds a confidence threshold of log(0.8).

        Args:
            base_ids: Token IDs of the prompt up to the function name position.

        Returns:
            The name of the most likely function to call.

        Raises:
            ValueError: If the best score is below log(0.8), meaning no
                function matched the prompt with sufficient confidence.
        """
        scoring_base = base_ids + self._fn_common_prefix_ids
        scores = [0.0] * len(self._fn_names)
        for depth, groups in enumerate(self._fn_name_groups):
            for prefix, idxs in groups.items():
                logits_arr = np.array(
                    self._llm.get_logits_from_input_ids(
                        scoring_base + list(prefix)))
                m = logits_arr.max()
                log_probs = logits_arr - m - \
                    np.log(np.sum(np.exp(logits_arr - m)))
                for i in idxs:
                    tid = self._fn_tids_short[i][depth]
                    scores[i] += float(log_probs[tid])
        best_idx = int(np.argmax(scores))
        n_tokens = len(self._fn_tids_short[best_idx])
        confidence = scores[best_idx] / n_tokens
        if confidence < np.log(0.8):
            raise ValueError("No function matched the prompt")
        self._last_confidence = confidence
        return self._fn_names[best_idx]

    def _decode_number(self, input_ids: list[int]) -> tuple[float, list[int]]:
        """Decodes a number parameter using constrained token selection.

        At each step, only tokens that extend a valid numeric prefix are
        allowed. Stops when the model selects a terminator after a complete
        number, or when no valid tokens remain.

        Args:
            input_ids: Current token ID sequence up to the parameter position.

        Returns:
            A tuple of (decoded float value, updated token ID sequence).

        Raises:
            ValueError: If the generated string cannot be parsed as a float.
        """
        generated = ""
        current_ids = list(input_ids)
        terminators = set(',} \n\t"')
        terminated = False
        for _ in range(self._MAX_STEPS["number"]):
            logits = self._llm.get_logits_from_input_ids(current_ids)
            logits_arr = np.array(logits)
            unconstrained_best = int(np.argmax(logits_arr))
            unconstrained_str = self._llm.decode([unconstrained_best])
            if (self._is_complete_number(generated)
                    and unconstrained_str
                    and unconstrained_str[0] in terminators):
                terminated = True
                break
            valid_ids = self._number_ids[
                [self._is_number_prefix(
                    generated + self._rev_vocab[int(tid)]
                ) for tid in self._number_ids]
            ]
            if len(valid_ids) == 0:
                terminated = True
                break
            best_id = int(valid_ids[np.argmax(logits_arr[valid_ids])])
            generated += self._rev_vocab[best_id]
            current_ids.append(best_id)
        if not terminated:
            raise ValueError(f"Number decode hit token limit: {generated!r}")
        try:
            return float(generated), current_ids
        except ValueError:
            raise ValueError(f"Failed to decode number: {generated!r}")

    def _decode_string(self, input_ids: list[int]) -> tuple[str, list[int]]:
        """Decodes a string parameter by generating tokens until a closing
        quote.

        Stops when an unescaped `"` or a newline is encountered. Raises if
        the token limit is reached without a proper termination.

        Args:
            input_ids: Current token ID sequence up to the parameter position.

        Returns:
            A tuple of (decoded string value stripped of whitespace,
            updated token ID sequence).

        Raises:
            ValueError: If the token limit is reached without finding a
                closing quote or newline.
        """
        generated = ""
        current_ids = list(input_ids)
        terminated = False
        for _ in range(self._MAX_STEPS["string"]):
            logits = self._llm.get_logits_from_input_ids(current_ids)
            best_id = int(np.argmax(logits))
            best_str = self._llm.decode([best_id])
            if '\n' in best_str:
                terminated = True
                break
            quote_idx = self._unescaped_quote_idx(best_str)
            if quote_idx is not None:
                generated += best_str[:quote_idx]
                current_ids.append(best_id)
                terminated = True
                break
            generated += best_str
            current_ids.append(best_id)
        if not terminated:
            raise ValueError(f"String decode hit token limit: {generated!r}")
        try:
            unescaped = json.loads('"' + generated + '"')
        except json.JSONDecodeError:
            unescaped = generated
        return unescaped.strip(), current_ids

    def _decode_boolean(self, input_ids: list[int]) -> tuple[bool, list[int]]:
        """Decodes a boolean parameter constrained to 'true' or 'false'.

        At each step, only tokens that extend a valid prefix of 'true' or
        'false' are allowed. Raises if a complete literal is not reached.

        Args:
            input_ids: Current token ID sequence up to the parameter position.

        Returns:
            A tuple of (decoded boolean value, updated token ID sequence).

        Raises:
            ValueError: If the token limit is reached without generating
                a complete 'true' or 'false' literal.
        """
        generated = ""
        current_ids = list(input_ids)
        literals = ("true", "false")
        for _ in range(self._MAX_STEPS["boolean"]):
            valid_ids = [
                tid for tid in self._bool_ids
                if any(
                    lit.startswith(generated + self._rev_vocab[tid])
                    for lit in literals
                )
            ]
            if not valid_ids:
                break
            logits = self._llm.get_logits_from_input_ids(current_ids)
            best_id = max(valid_ids, key=lambda tid: logits[tid])
            generated += self._rev_vocab[best_id]
            current_ids.append(best_id)
            if generated in literals:
                break
        if generated not in ("true", "false"):
            raise ValueError(f"Boolean decode hit token limit: {generated!r}")
        return generated == "true", current_ids

    def choose_function_call(self, prompt: str) -> FunctionCall:
        """Translates a natural language prompt into a structured function
        call.

        Selects the best matching function using log-probability scoring, then
        decodes each parameter using type-aware constrained decoding.

        Args:
            prompt: The natural language request to process.

        Returns:
            A FunctionCall containing the function name and decoded parameters.

        Raises:
            ValueError: If any parameter cannot be decoded within its token
            limit.
        """
        if self._show:
            print(f"Prompt: {prompt}")
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
            value: str | float | int | bool
            if param_type in ("number", "integer"):
                value, input_ids = self._decode_number(input_ids)
                if param_type == "integer":
                    value = int(value)
            elif param_type == "boolean":
                value, input_ids = self._decode_boolean(input_ids)
            else:
                value, input_ids = self._decode_string(input_ids)
                if isinstance(value, str) and (
                    value in (param_type, param_name)
                    or value.strip("'\"") == ""
                    or (
                        ("''" in prompt or '""' in prompt)
                        and len(value.split()) >= 3
                        and set(
                            ''.join(
                                c if c.isalnum() else ' ' for c in value
                            ).lower().split()
                        ) <= set(
                            ''.join(
                                c if c.isalnum() else ' ' for c in prompt
                            ).lower().split()
                        )
                    )
                ):
                    value = ""
            parameters[param_name] = value

        result = FunctionCall(name=fn_name, parameters=parameters)
        if self._show:
            print(f"  -> {fn_name}{parameters} "
                  f"(confidence: {self._last_confidence:.3f})")
        return result
