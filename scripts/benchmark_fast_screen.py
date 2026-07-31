from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from codegap_qa.fast_backend import diagnostics, screen_qc_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=48400)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    rng = np.random.default_rng(20260723)
    a = np.vstack(
        [np.sort(rng.choice(12, 3, replace=False)) for _ in range(args.candidates)]
    ).astype(np.uint16)
    b = np.vstack(
        [np.sort(rng.choice(12, 3, replace=False)) for _ in range(args.candidates)]
    ).astype(np.uint16)
    backend_counts: dict[str, int] = {}
    started = perf_counter()
    passed = 0
    for begin in range(0, args.candidates, args.batch_size):
        end = min(args.candidates, begin + args.batch_size)
        result, status = screen_qc_batch(
            a[begin:end],
            b[begin:end],
            l=3,
            m=4,
            min_dx=4,
            min_dz=4,
            min_k=2,
            backend=args.backend,
        )
        backend_counts[status.selected] = backend_counts.get(status.selected, 0) + end - begin
        passed += int(np.sum((result["flags"] & 0x07) == 0x07))
    elapsed = perf_counter() - started
    print(json.dumps({
        "candidates": args.candidates,
        "elapsed_seconds": elapsed,
        "candidates_per_second": args.candidates / elapsed,
        "passed": passed,
        "backend_counts": backend_counts,
        "diagnostics": diagnostics(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
