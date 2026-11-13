"""python -m evalkit demo                       run the bundled demo
   python -m evalkit compare base.json new.json --scorer NAME [--alpha 0.05]
       exit code 1 if the new run is a statistically significant regression (for CI)"""
from __future__ import annotations

import argparse
import sys

from .runner import Run
from .stats import compare_runs


def main() -> int:
    parser = argparse.ArgumentParser(prog="evalkit")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    cmp = sub.add_parser("compare")
    cmp.add_argument("baseline")
    cmp.add_argument("candidate")
    cmp.add_argument("--scorer", required=True)
    cmp.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if args.command == "compare":
        result = compare_runs(Run.load(args.baseline), Run.load(args.candidate), args.scorer)
        print(f"{args.scorer}: {result.baseline_mean:.3f} -> {result.candidate_mean:.3f} ({result.delta:+.3f}), "
              f"fixed {len(result.fixed)}, broke {len(result.broken)}, p = {result.p_value:.3g}")
        if result.broken:
            print("newly failing:", ", ".join(result.broken[:20]))
        if result.is_regression(args.alpha):
            print("REGRESSION: the drop is statistically significant")
            return 1
        return 0
    from .demo import main as demo_main
    return demo_main()


if __name__ == "__main__":
    sys.exit(main())
