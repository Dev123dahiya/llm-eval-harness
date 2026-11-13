"""Scorers turn (output, expected) into a number in [0, 1] plus a reason a human can read."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Score:
    value: float
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.value >= 1.0


class Scorer(Protocol):
    name: str

    def __call__(self, output: str, expected: Any) -> Score: ...


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


class ExactMatch:
    name = "exact_match"

    def __init__(self, normalise: bool = True) -> None:
        self.normalise = normalise

    def __call__(self, output: str, expected: Any) -> Score:
        a, b = (_normalise(output), _normalise(expected)) if self.normalise else (output, str(expected))
        return Score(float(a == b), "" if a == b else f"expected {expected!r}, got {output!r}")


class Contains:
    name = "contains"

    def __call__(self, output: str, expected: Any) -> Score:
        wanted = expected if isinstance(expected, list) else [expected]
        missing = [w for w in wanted if _normalise(w) not in _normalise(output)]
        return Score(1 - len(missing) / len(wanted), f"missing {missing}" if missing else "")


class RegexMatch:
    name = "regex"

    def __init__(self, pattern: str | None = None) -> None:
        self.pattern = pattern

    def __call__(self, output: str, expected: Any) -> Score:
        pattern = self.pattern or str(expected)
        ok = re.search(pattern, output) is not None
        return Score(float(ok), "" if ok else f"no match for /{pattern}/")


class NumericClose:
    """First number in the output is within a tolerance of the expected value."""

    name = "numeric_close"

    def __init__(self, rel_tol: float = 0.0, abs_tol: float = 1e-9) -> None:
        self.rel_tol, self.abs_tol = rel_tol, abs_tol

    def __call__(self, output: str, expected: Any) -> Score:
        match = re.search(r"-?\d[\d,]*\.?\d*", output)
        if not match:
            return Score(0.0, "no number in output")
        got = float(match.group().replace(",", ""))
        ok = abs(got - float(expected)) <= max(self.abs_tol, self.rel_tol * abs(float(expected)))
        return Score(float(ok), "" if ok else f"expected {expected}, got {got}")


class TokenF1:
    """Word-overlap F1, the SQuAD metric: partial credit for partially right free text."""

    name = "token_f1"

    def __call__(self, output: str, expected: Any) -> Score:
        got, want = _normalise(output).split(), _normalise(expected).split()
        if not got or not want:
            return Score(float(got == want))
        overlap = sum((Counter(got) & Counter(want)).values())
        if overlap == 0:
            return Score(0.0, "no shared words")
        precision, recall = overlap / len(got), overlap / len(want)
        return Score(2 * precision * recall / (precision + recall))


def _extract_json(output: str) -> Any:
    """Parse JSON, tolerating a Markdown code fence around it (models add those often)."""
    text = output.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    return json.loads(fenced.group(1) if fenced else text)


class JsonValid:
    """Output parses as JSON and has the required keys with the right types."""

    name = "json_valid"

    def __init__(self, required: dict[str, type | tuple[type, ...]] | None = None) -> None:
        self.required = required or {}

    def __call__(self, output: str, expected: Any = None) -> Score:
        try:
            data = _extract_json(output)
        except (json.JSONDecodeError, TypeError) as exc:
            return Score(0.0, f"invalid JSON: {exc}")
        if self.required and not isinstance(data, dict):
            return Score(0.0, "expected a JSON object")
        for key, kind in self.required.items():
            if key not in data:
                return Score(0.0, f"missing key {key!r}")
            if not isinstance(data[key], kind) or isinstance(data[key], bool) and kind is not bool:
                return Score(0.0, f"{key!r} has type {type(data[key]).__name__}")
        return Score(1.0)


class JsonFields:
    """Fraction of expected fields the output got right. Invalid JSON scores 0."""

    name = "json_fields"

    def __call__(self, output: str, expected: Any) -> Score:
        try:
            data = _extract_json(output)
        except (json.JSONDecodeError, TypeError):
            return Score(0.0, "invalid JSON")
        if not isinstance(data, dict):
            return Score(0.0, "expected a JSON object")
        wrong = [k for k, v in expected.items() if data.get(k) != v]
        return Score(1 - len(wrong) / len(expected), f"wrong fields: {wrong}" if wrong else "")


class LLMJudge:
    """Grade with a model against a rubric. The judge is any callable prompt -> text.

    Judges are noisy and can be wrong, so the prompt asks for a fixed verdict
    format and anything else is scored 0 with the raw reply kept as the reason.
    Validate a judge against human labels before trusting its numbers.
    """

    name = "llm_judge"
    TEMPLATE = ("You are grading an answer.\nRubric: {rubric}\n\nReference: {expected}\n\nAnswer to grade:\n{output}\n\n"
                "Reply with one line: VERDICT: PASS or VERDICT: FAIL, then a short reason.")

    def __init__(self, judge: Callable[[str], str], rubric: str) -> None:
        self.judge, self.rubric = judge, rubric

    def __call__(self, output: str, expected: Any) -> Score:
        reply = self.judge(self.TEMPLATE.format(rubric=self.rubric, expected=expected, output=output))
        match = re.search(r"VERDICT:\s*(PASS|FAIL)", reply, flags=re.I)
        if not match:
            return Score(0.0, f"unparseable judge reply: {reply[:80]!r}")
        return Score(float(match.group(1).upper() == "PASS"), reply.strip()[:200])
