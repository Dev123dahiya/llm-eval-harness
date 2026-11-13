"""Evaluate text-generating systems: score outputs, quantify uncertainty, gate regressions."""
from .scorers import (Contains, ExactMatch, JsonFields, JsonValid, LLMJudge, NumericClose, RegexMatch, Score, TokenF1)
from .runner import Case, CaseResult, Run, load_cases, run_eval
from .stats import bootstrap_ci, compare_runs, mcnemar_exact

__all__ = ["Contains", "ExactMatch", "JsonFields", "JsonValid", "LLMJudge", "NumericClose", "RegexMatch", "Score",
           "TokenF1", "Case", "CaseResult", "Run", "load_cases", "run_eval", "bootstrap_ci", "compare_runs", "mcnemar_exact"]
