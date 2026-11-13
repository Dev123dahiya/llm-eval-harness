import json
import math
import threading

import pytest

from evalkit import (Case, Contains, ExactMatch, JsonFields, JsonValid, LLMJudge, NumericClose, RegexMatch, Run,
                     TokenF1, bootstrap_ci, compare_runs, load_cases, mcnemar_exact, run_eval)
from evalkit.__main__ import main as cli
from evalkit.demo import DATA, extractor_v1, extractor_v2


def test_text_scorers():
    assert ExactMatch()("  Paris. ", "paris").passed
    assert not ExactMatch(normalise=False)("Paris.", "paris").passed
    assert Contains()("The capital is Paris, France", ["paris", "france"]).value == 1.0
    partial = Contains()("The capital is Paris", ["paris", "france"])
    assert partial.value == 0.5 and "france" in partial.reason
    assert RegexMatch(r"^\d{4}-\d{2}-\d{2}$")("2026-03-14", None).passed
    assert not RegexMatch()("hello", r"\d+").passed


def test_numeric_and_f1_scorers():
    assert NumericClose()("The total is 1,250.50 rupees", 1250.5).passed
    assert NumericClose(rel_tol=0.01)("about 101", 100).passed
    assert not NumericClose()("about 101", 100).passed
    assert NumericClose()("no digits here", 5).reason == "no number in output"
    assert TokenF1()("the cat sat", "the cat sat").value == 1.0
    assert TokenF1()("the cat", "the cat sat down").value == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert TokenF1()("dogs bark", "cats meow").value == 0.0


def test_json_scorers_tolerate_fences_and_catch_type_errors():
    schema = JsonValid({"amount": int, "currency": str})
    assert schema('{"amount": 5, "currency": "INR"}').passed
    assert schema('```json\n{"amount": 5, "currency": "INR"}\n```').passed
    assert "invalid JSON" in schema("amount is 5").reason
    assert "missing key" in schema('{"amount": 5}').reason
    assert "has type" in schema('{"amount": "5", "currency": "INR"}').reason
    assert not schema('{"amount": true, "currency": "INR"}').passed   # bool is not an int here
    assert not schema("[1, 2]").passed
    fields = JsonFields()('{"a": 1, "b": 2, "c": 9}', {"a": 1, "b": 2, "c": 3})
    assert fields.value == pytest.approx(2 / 3) and fields.reason == "wrong fields: ['c']"
    assert JsonFields()("not json", {"a": 1}).value == 0.0


def test_llm_judge_parses_verdicts_and_fails_closed():
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        return "VERDICT: PASS - matches the reference"

    scorer = LLMJudge(judge, rubric="Is the answer factually consistent with the reference?")
    assert scorer("Paris", "Paris is the capital").passed
    assert "Rubric: Is the answer factually consistent" in prompts[0] and "Paris is the capital" in prompts[0]
    assert not LLMJudge(lambda p: "verdict: fail, wrong city", "r")("Lyon", "Paris").passed
    rambling = LLMJudge(lambda p: "Well, it depends on how you look at it.", "r")("x", "y")
    assert rambling.value == 0.0 and "unparseable" in rambling.reason


CASES = [Case(f"c{i}", str(i), str(i * 2), ("even" if i % 2 == 0 else "odd",)) for i in range(20)]


def test_runner_keeps_order_scores_and_groups_by_tag():
    run = run_eval("double", lambda x: str(int(x) * 2 if int(x) % 2 == 0 else 0), CASES, [ExactMatch()], workers=8)
    assert [r.case_id for r in run.results] == [c.id for c in CASES]
    assert run.mean("exact_match") == 0.5
    assert run.by_tag("exact_match") == {"even": 1.0, "odd": 0.0}
    assert len(run.failures("exact_match")) == 10 and "expected" in run.failures("exact_match")[0].reasons["exact_match"]


def test_runner_retries_with_backoff_then_records_the_error():
    attempts, sleeps, lock = {}, [], threading.Lock()

    def flaky(x):
        with lock:
            attempts[x] = attempts.get(x, 0) + 1
            n = attempts[x]
        if x == "always":
            raise TimeoutError("upstream timed out")
        if x == "once" and n == 1:
            raise ConnectionError("reset")
        return "ok"

    cases = [Case("a", "fine", "ok"), Case("b", "once", "ok"), Case("c", "always", "ok")]
    run = run_eval("flaky", flaky, cases, [ExactMatch()], workers=1, retries=2, backoff_s=1.0, sleep=sleeps.append)
    a, b, c = run.results
    assert (a.attempts, b.attempts, c.attempts) == (1, 2, 3)
    assert b.scores["exact_match"] == 1.0 and b.error is None
    assert c.scores["exact_match"] == 0.0 and c.error == "TimeoutError: upstream timed out"
    assert "system error" in c.reasons["exact_match"]
    assert sleeps == [1.0, 1.0, 2.0]                 # b waits once; c backs off 1 s then 2 s, none after the last try
    assert run.mean("exact_match") == pytest.approx(2 / 3)  # the failed case still counts in the denominator


def test_cache_skips_the_system_on_a_second_run_and_is_keyed_by_version(tmp_path):
    calls = []

    def system(x):
        calls.append(x)
        return x.upper()

    cases = [Case("a", "hello", "HELLO"), Case("b", "world", "WORLD")]
    first = run_eval("v1", system, cases, [ExactMatch()], cache_dir=str(tmp_path))
    second = run_eval("v1", system, cases, [Contains()], cache_dir=str(tmp_path))   # new scorer, no new calls
    assert len(calls) == 2 and all(r.cached for r in second.results) and not any(r.cached for r in first.results)
    assert second.mean("contains") == 1.0
    run_eval("v2", system, cases, [ExactMatch()], cache_dir=str(tmp_path))          # new version must not reuse v1 outputs
    assert len(calls) == 4


def test_load_cases_and_run_round_trip(tmp_path):
    cases = load_cases(DATA)
    assert len(cases) == 80 and cases[0].tags == ("plain",) and isinstance(cases[0].expected, dict)
    run = run_eval("v2", extractor_v2, cases[:10], [JsonFields()])
    path = str(tmp_path / "run.json")
    run.save(path)
    loaded = Run.load(path)
    assert loaded.name == "v2" and loaded.mean("json_fields") == run.mean("json_fields") and loaded.results[0].tags == ("plain",)
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"id": "x", "input": "1"}\n{"id": "x", "input": "2"}\n')
    with pytest.raises(ValueError, match="unique"):
        load_cases(str(dup))


def test_bootstrap_interval_brackets_the_mean_and_narrows_with_more_data():
    small = [1.0] * 8 + [0.0] * 2
    lo, hi = bootstrap_ci(small)
    assert lo < 0.8 < hi and 0.4 <= lo and hi <= 1.0
    lo_big, hi_big = bootstrap_ci(small * 50)
    assert (hi_big - lo_big) < (hi - lo) / 3
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    assert bootstrap_ci(small) == bootstrap_ci(small)       # seeded, so reproducible
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 * (1 / 32))       # 0.0625: five wins in a row is not yet significant
    assert mcnemar_exact(0, 6) == pytest.approx(2 * (1 / 64))       # 0.03125: six is
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 1024)
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)


def make_run(name, passed):
    from evalkit import CaseResult
    return Run(name, [CaseResult(f"c{i}", "", {"m": float(p)}, {"m": ""}, 0.0, 1) for i, p in enumerate(passed)])


def test_compare_runs_separates_noise_from_real_regressions():
    base = make_run("base", [1] * 40)
    noise = compare_runs(base, make_run("cand", [1] * 38 + [0] * 2), "m")
    assert noise.delta < 0 and noise.broken == ["c38", "c39"] and not noise.is_regression()   # p = 0.5
    real = compare_runs(base, make_run("cand", [1] * 30 + [0] * 10), "m")
    assert real.is_regression() and real.p_value < 0.01 and real.diff_ci[1] < 0
    better = compare_runs(make_run("b", [0] * 10 + [1] * 30), base, "m")
    assert len(better.fixed) == 10 and not better.is_regression()
    with pytest.raises(ValueError):
        compare_runs(base, Run("empty"), "m")


def test_demo_shows_an_overall_win_hiding_a_group_regression():
    cases = load_cases(DATA)
    v1 = run_eval("v1", extractor_v1, cases, [JsonFields()])
    v2 = run_eval("v2", extractor_v2, cases, [JsonFields()])
    assert v2.mean("json_fields") > v1.mean("json_fields")
    assert v2.by_tag("json_fields")["symbol_after"] < v1.by_tag("json_fields")["symbol_after"]
    result = compare_runs(v1, v2, "json_fields")
    assert len(result.fixed) == 48 and len(result.broken) == 16


def test_cli_compare_exit_code_gates_ci(tmp_path, monkeypatch, capsys):
    base, bad = str(tmp_path / "base.json"), str(tmp_path / "bad.json")
    make_run("base", [1] * 40).save(base)
    make_run("bad", [1] * 28 + [0] * 12).save(bad)
    monkeypatch.setattr("sys.argv", ["evalkit", "compare", base, bad, "--scorer", "m"])
    assert cli() == 1 and "REGRESSION" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["evalkit", "compare", bad, base, "--scorer", "m"])
    assert cli() == 0
