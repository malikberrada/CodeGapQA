from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .fast_backend import diagnostics as fast_diagnostics
from .freeze import freeze, verify_freeze
from .optimizer import search_code_families
from .research_triad import run_research_triad  # CODEGAP_V090_RESEARCH_TRIAD
from .pipeline import load_config, run_pipeline
from .qpu_cli import add_qpu_commands, handle_qpu_command
from .recovery import recover_verifier_from_artifacts
from .promotion import promote_recovered_artifact


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="codegap")
    commands = root.add_subparsers(dest="command", required=True)

    pipeline = commands.add_parser("pipeline")
    pipeline.add_argument("--config", type=Path, required=True)
    pipeline.add_argument("--out", type=Path, required=True)
    pipeline.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable all tqdm progress bars.",
    )

    search = commands.add_parser("search")
    search.add_argument("--config", type=Path, required=True)
    search.add_argument("--out", type=Path, required=True)
    search.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable all tqdm progress bars.",
    )

    # CODEGAP_V090_RESEARCH_TRIAD
    triad = commands.add_parser(
        "research-triad",
        help=(
            "Jointly search code families, registered hardware-noise "
            "robustness, and low-qubit verification/simulation gap."
        ),
    )
    triad.add_argument("--config", type=Path, required=True)
    triad.add_argument("--out", type=Path, required=True)
    triad.add_argument(
        "--mode",
        choices=("proxy", "deep"),
        default=None,
        help="Override research_triad.mode from the JSON configuration.",
    )
    triad.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable all tqdm progress bars.",
    )

    freeze_command = commands.add_parser("freeze")
    freeze_command.add_argument("--root", type=Path, required=True)

    verify = commands.add_parser("verify-freeze")
    verify.add_argument("--manifest", type=Path, required=True)

    commands.add_parser(
        "fast-info",
        help="Show C++/OpenMP and CUDA acceleration availability.",
    )

    recover = commands.add_parser(
        "recover-verifier",
        help="Recover local verifier observables from existing v0.5.2 artifacts without rerunning Cotengra.",
    )
    recover.add_argument("--artifact", type=Path, required=True)
    recover.add_argument("--config", type=Path, required=True)
    recover.add_argument("--out", type=Path, required=True)
    recover.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Candidate ID to reoptimize. Repeat to define primary and backup "
            "priority order."
        ),
    )

    promote = commands.add_parser(
        "promote-recovery",
        help=(
            "Promote a successful verifier recovery into a frozen artifact "
            "authorized for fresh QPU-native preparation."
        ),
    )
    promote.add_argument("--artifact", type=Path, required=True)
    promote.add_argument("--recovery", type=Path, required=True)
    promote.add_argument("--out", type=Path, required=True)

    add_qpu_commands(commands)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "pipeline":
        if args.no_progress:
            os.environ["CODEGAP_PROGRESS"] = "0"
        result = run_pipeline(args.config, args.out)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] != "STOP_NO_CODE_CANDIDATE" else 2
    if args.command == "search":
        if args.no_progress:
            os.environ["CODEGAP_PROGRESS"] = "0"
        config = load_config(args.config)
        frontier, certificate = search_code_families(config, args.out)
        print(
            json.dumps(
                {
                    "frontier_size": len(frontier),
                    "certificate": certificate,
                },
                indent=2,
            )
        )
        return 0 if frontier else 2
    if args.command == "research-triad":  # CODEGAP_V090_RESEARCH_TRIAD
        if args.no_progress:
            os.environ["CODEGAP_PROGRESS"] = "0"
        config = load_config(args.config)
        if args.mode is not None:
            config.setdefault("research_triad", {})["mode"] = args.mode
        result = run_research_triad(config, args.out)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "freeze":
        print(json.dumps(freeze(args.root), indent=2))
        return 0
    if args.command == "verify-freeze":
        result = verify_freeze(args.manifest)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2
    if args.command == "fast-info":
        print(json.dumps(fast_diagnostics(), indent=2))
        return 0
    if args.command == "recover-verifier":
        config = load_config(args.config)
        result = recover_verifier_from_artifacts(
            args.artifact, config, args.out, candidate_ids=args.candidate
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "RECOVERED_CANDIDATE_FOUND" else 2
    if args.command == "promote-recovery":
        result = promote_recovered_artifact(
            artifact=args.artifact,
            recovery=args.recovery,
            output=args.out,
        )
        print(json.dumps(result, indent=2))
        return 0 if result.get("freeze_verified") else 2
    qpu_result = handle_qpu_command(args)
    if qpu_result is not None:
        return qpu_result
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
