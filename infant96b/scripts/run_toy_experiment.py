#!/usr/bin/env python3
import argparse
import os
import sys
import time


def _parse_args():
    parser = argparse.ArgumentParser(description="Run toy neuroplasticity experiments.")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--k_out", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)

    sleep_group = parser.add_mutually_exclusive_group()
    sleep_group.add_argument("--sleep", dest="sleep", action="store_true")
    sleep_group.add_argument("--no-sleep", dest="sleep", action="store_false")
    parser.set_defaults(sleep=None)

    rewire_group = parser.add_mutually_exclusive_group()
    rewire_group.add_argument("--rewire", dest="rewire", action="store_true")
    rewire_group.add_argument("--no-rewire", dest="rewire", action="store_false")
    parser.set_defaults(rewire=None)

    return parser.parse_args()


def main():
    args = _parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from train.toy.run_toy_experiment import (
        Condition,
        ExperimentConfig,
        default_conditions,
        run_conditions,
    )

    cfg = ExperimentConfig(
        N=args.N,
        d=args.d,
        k_out=args.k_out,
        steps=args.steps,
        seed=args.seed,
    )

    output_dir = os.path.join(root, "runs")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"toy_experiment_{timestamp}.csv")

    if args.sleep is None:
        conditions = default_conditions()
    else:
        sleep_enabled = bool(args.sleep)
        if sleep_enabled:
            rewire_enabled = True if args.rewire is None else bool(args.rewire)
            name = "sleep" if rewire_enabled else "sleep_no_rewire"
            conditions = [
                Condition(name="no_sleep", sleep_enabled=False, rewire_enabled=False),
                Condition(name=name, sleep_enabled=True, rewire_enabled=rewire_enabled),
            ]
        else:
            conditions = [
                Condition(name="no_sleep", sleep_enabled=False, rewire_enabled=False)
            ]

    results = run_conditions(cfg, conditions, output_csv)

    print(f"CSV saved to: {output_csv}")
    for row in results:
        print(
            "condition={condition} acc_taskA_post_B={acc_taskA_post_B:.3f} "
            "drift_risk={drift_risk:.4f} synaptic_load={synaptic_load:.4f}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
