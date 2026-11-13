from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .scorers import Score, Scorer


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    expected: Any = None
    tags: tuple[str, ...] = ()


@dataclass
class CaseResult:
    case_id: str
    output: str
    scores: dict[str, float]
    reasons: dict[str, str]
    latency_s: float
    attempts: int
    error: str | None = None
    cached: bool = False
    tags: tuple[str, ...] = ()


@dataclass
class Run:
    name: str
    results: list[CaseResult] = field(default_factory=list)

    def values(self, scorer: str, tag: str | None = None) -> list[float]:
        return [r.scores[scorer] for r in self.results if tag is None or tag in r.tags]

    def mean(self, scorer: str, tag: str | None = None) -> float:
        v = self.values(scorer, tag)
        return sum(v) / len(v) if v else float("nan")

    def by_tag(self, scorer: str) -> dict[str, float]:
        tags = sorted({t for r in self.results for t in r.tags})
        return {t: self.mean(scorer, t) for t in tags}

    def failures(self, scorer: str) -> list[CaseResult]:
        return [r for r in self.results if r.scores[scorer] < 1.0]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "results": [asdict(r) for r in self.results]}, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Run":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["name"], [CaseResult(**{**r, "tags": tuple(r["tags"])}) for r in data["results"]])


def load_cases(path: str) -> list[Case]:
    """Read cases from JSON Lines: one {"id", "input", "expected", "tags"} object per line."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                cases.append(Case(d["id"], d["input"], d.get("expected"), tuple(d.get("tags", ()))))
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("case ids must be unique")
    return cases


class OutputCache:
    """Disk cache keyed by system version and input, so re-scoring never re-pays for generation."""

    def __init__(self, directory: str | None) -> None:
        self.directory = directory
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _path(self, version: str, text: str) -> str:
        key = hashlib.sha256(f"{version}\x00{text}".encode()).hexdigest()
        return os.path.join(self.directory, f"{key}.json")

    def get(self, version: str, text: str) -> str | None:
        if not self.directory:
            return None
        try:
            with open(self._path(version, text), encoding="utf-8") as f:
                return json.load(f)["output"]
        except FileNotFoundError:
            return None

    def put(self, version: str, text: str, output: str) -> None:
        if self.directory:
            with open(self._path(version, text), "w", encoding="utf-8") as f:
                json.dump({"output": output}, f)


def run_eval(
    name: str,
    system: Callable[[str], str],
    cases: list[Case],
    scorers: list[Scorer],
    *,
    workers: int = 4,
    retries: int = 2,
    backoff_s: float = 0.5,
    cache_dir: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Run:
    """Run every case through the system and score it.

    Cases run concurrently because the system under test is usually a network
    call. A case whose call keeps failing is recorded with score 0 and its
    error, so one bad case cannot abort or silently shrink the run.
    """
    cache = OutputCache(cache_dir)

    def one(case: Case) -> CaseResult:
        cached = cache.get(name, case.input)
        output, error, attempts, start = cached, None, 0, time.perf_counter()
        while output is None and attempts <= retries:
            attempts += 1
            try:
                output = str(system(case.input))
                cache.put(name, case.input, output)
            except Exception as exc:  # noqa: BLE001 - any failure of the system under test is data
                error = f"{type(exc).__name__}: {exc}"
                if attempts <= retries:
                    sleep(backoff_s * 2 ** (attempts - 1))
        latency = time.perf_counter() - start
        if output is None:
            return CaseResult(case.id, "", {s.name: 0.0 for s in scorers}, {s.name: f"system error: {error}" for s in scorers},
                              latency, attempts, error, tags=case.tags)
        scored: dict[str, Score] = {s.name: s(output, case.expected) for s in scorers}
        return CaseResult(case.id, output, {k: v.value for k, v in scored.items()}, {k: v.reason for k, v in scored.items()},
                          latency, attempts, cached=cached is not None, tags=case.tags)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, cases))  # map preserves case order
    return Run(name, results)
