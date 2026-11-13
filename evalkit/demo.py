"""A self-contained demo with no model involved.

Two versions of a rule-based invoice extractor stand in for "prompt v1" and
"prompt v2". They are deterministic, so the demo's numbers are reproducible,
and they fail in different ways, which is what the harness is built to show.
"""
from __future__ import annotations

import json
import os
import random
import re

from .runner import Case, load_cases, run_eval
from .scorers import JsonFields, JsonValid
from .stats import bootstrap_ci, compare_runs

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "extraction.jsonl")
SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
WORDS = {"rupees": "INR", "dollars": "USD", "euros": "EUR", "pounds": "GBP"}


def generate_cases(n: int = 80, seed: int = 4) -> list[dict]:
    """Synthetic invoice sentences in five groups, each stressing a different thing."""
    rng = random.Random(seed)
    vendors = ["Acme Ltd", "Northgrid Power", "Blue Kite Studio", "Orbit Freight", "Sagar Traders"]
    cases = []
    for i in range(n):
        tag = ["plain", "separators", "currency_word", "distractor", "symbol_after"][i % 5]
        invoice = f"INV-{rng.randint(100, 9999)}"
        symbol, code = rng.choice(list(SYMBOLS.items()))
        amount = rng.randint(10, 999) if tag in ("plain", "symbol_after") else rng.randint(1_000, 250_000)
        vendor = rng.choice(vendors)
        if tag == "plain":
            text = f"Invoice {invoice} from {vendor} totals {symbol}{amount}."
        elif tag == "symbol_after":
            text = f"Invoice {invoice} from {vendor} totals {amount} {symbol}."
        elif tag == "separators":
            text = f"Invoice {invoice} from {vendor} totals {symbol}{amount:,}.00, payable on receipt."
        elif tag == "currency_word":
            word = next(w for w, c in WORDS.items() if c == code)
            text = f"{vendor} billed {amount:,} {word} on invoice {invoice}."
        else:
            text = f"Invoice {invoice}: {rng.randint(2, 40)} items from {vendor}, net {rng.randint(30, 90)} days, total {symbol}{amount:,}."
        cases.append({"id": f"case-{i:03d}", "input": text, "tags": [tag],
                      "expected": {"invoice_id": invoice, "amount": amount, "currency": code}})
    return cases


def extractor_v1(text: str) -> str:
    """Naive: first run of digits after the invoice id is the amount; symbols only."""
    invoice = re.search(r"INV-\d+", text)
    rest = text.replace(invoice.group(), "") if invoice else text
    number = re.search(r"\d+", rest)
    currency = next((code for symbol, code in SYMBOLS.items() if symbol in text), None)
    return json.dumps({"invoice_id": invoice.group() if invoice else None,
                       "amount": int(number.group()) if number else None, "currency": currency})


def extractor_v2(text: str) -> str:
    """Better on hard inputs: the amount is the number that follows a currency symbol or precedes a currency word.

    It has its own blind spot: it expects the symbol before the number, so "115 €" is missed.
    """
    invoice = re.search(r"INV-\d+", text)
    amount = currency = None
    by_symbol = re.search(r"([₹$€£])\s*([\d,]+(?:\.\d+)?)", text)
    by_word = re.search(r"([\d,]+(?:\.\d+)?)\s*(rupees|dollars|euros|pounds)", text)
    if by_symbol:
        currency, amount = SYMBOLS[by_symbol.group(1)], by_symbol.group(2)
    elif by_word:
        currency, amount = WORDS[by_word.group(2)], by_word.group(1)
    value = int(float(amount.replace(",", ""))) if amount else None
    # Wraps the JSON in a code fence for one vendor, as chatty models do. The JSON scorers tolerate that.
    body = json.dumps({"invoice_id": invoice.group() if invoice else None, "amount": value, "currency": currency})
    return f"```json\n{body}\n```" if "Blue Kite" in text else body


def write_dataset() -> None:
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as f:
        for case in generate_cases():
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def main() -> int:
    cases: list[Case] = load_cases(DATA)
    scorers = [JsonValid({"invoice_id": str, "amount": int, "currency": str}), JsonFields()]
    v1 = run_eval("extractor-v1", extractor_v1, cases, scorers)
    v2 = run_eval("extractor-v2", extractor_v2, cases, scorers)

    print(f"{len(cases)} cases. Scorer: json_fields (share of fields correct), 95% bootstrap interval\n")
    print("| System | Overall | 95% CI | plain | separators | currency_word | distractor | symbol_after | Valid JSON |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for run in (v1, v2):
        lo, hi = bootstrap_ci(run.values("json_fields"))
        tags = run.by_tag("json_fields")
        print(f"| {run.name} | {run.mean('json_fields'):.3f} | {lo:.3f} to {hi:.3f} | {tags['plain']:.3f} | {tags['separators']:.3f} | "
              f"{tags['currency_word']:.3f} | {tags['distractor']:.3f} | {tags['symbol_after']:.3f} | {run.mean('json_valid'):.3f} |")

    cmp = compare_runs(v1, v2, "json_fields")
    print(f"\nv2 vs v1 on fully-correct cases: fixed {len(cmp.fixed)}, broke {len(cmp.broken)}, "
          f"McNemar exact p = {cmp.p_value:.2g}, mean difference {cmp.delta:+.3f} (95% CI {cmp.diff_ci[0]:+.3f} to {cmp.diff_ci[1]:+.3f})")
    for run in (v1, v2):
        example = run.failures("json_fields")[0]
        print(f"\nExample {run.name} failure [{example.case_id}]: {example.reasons['json_fields']}\n  input:  {cases[int(example.case_id[-3:])].input}\n  output: {example.output}")
    gate = compare_runs(v2, v1, "json_fields")  # pretend v1 is a proposed change on top of v2
    print(f"\nRegression gate, if someone proposed going back to v1: is_regression = {gate.is_regression()} (p = {gate.p_value:.2g})")
    return 0
