#!/usr/bin/env python3
"""Verify the sanitized public CodeGapQA release without provider access."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public_release"
SUMMARY_PATH = PUBLIC / "data" / "public_certificate_summary.json"
METRICS_PATH = PUBLIC / "data" / "fold_metrics.csv"
MANIFEST_PATH = PUBLIC / "SHA256SUMS.txt"

EXPECTED_ALIASES = {
    1: "confirmatory-fold-1",
    3: "confirmatory-fold-3",
    5: "confirmatory-fold-5",
}
EXPECTED_CZ = {1: 360, 3: 1080, 5: 1800}
EXPECTED_RX = {1: 660, 3: 1740, 5: 2820}
EXPECTED_RZ = {1: 1320, 3: 2760, 5: 4200}
EXPECTED_BARRIERS = {1: 0, 3: 360, 5: 720}
EXPECTED_MEASUREMENTS = {1: 60, 3: 60, 5: 60}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_qasm(path: Path) -> dict[str, int]:
    lines = [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        "rx": sum(line.startswith("rx(") for line in lines),
        "rz": sum(line.startswith("rz(") for line in lines),
        "cz": sum(line.startswith("cz ") for line in lines),
        "barrier": sum(line.startswith("barrier") for line in lines),
        "measure": sum("= measure " in line for line in lines),
    }


def verify_manifest() -> None:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing manifest: {MANIFEST_PATH}")
    for line_no, line in enumerate(MANIFEST_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise RuntimeError(f"Malformed manifest line {line_no}: {line!r}")
        expected, relative = match.groups()
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Manifest file missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )


def main() -> int:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["status"] != "QPU_PANEL_PASS":
        raise RuntimeError("Public certificate status is not QPU_PANEL_PASS")
    if summary["public_candidate_name"] != "codegapqa-60q-v1":
        raise RuntimeError("Unexpected public candidate name")
    if summary["folding_factors"] != [1, 3, 5]:
        raise RuntimeError("Unexpected folding factors")
    if int(summary["science_shots_per_fold"]) != 10000:
        raise RuntimeError("Science shot count changed")
    if float(summary["base_margin_lcb"]) <= 0:
        raise RuntimeError("Base-fold margin is not positive")
    if float(summary["minimum_margin_lcb"]) <= 0:
        raise RuntimeError("Minimum margin is not positive")
    if int(summary["folds_passed"]) != 3:
        raise RuntimeError("Not all three folds passed")
    if float(summary["gamma_log10"]) < float(summary["gamma_target"]):
        raise RuntimeError("Registered gap target is not met")

    public_circuits = {
        int(row["folding_factor"]): row["name"] for row in summary["public_circuits"]
    }
    if public_circuits != EXPECTED_ALIASES:
        raise RuntimeError(f"Unexpected public aliases: {public_circuits}")

    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        metrics = {int(row["fold"]): row for row in csv.DictReader(handle)}
    if set(metrics) != set(EXPECTED_ALIASES):
        raise RuntimeError("Metrics do not contain exactly folds 1, 3, and 5")
    for fold, row in metrics.items():
        if int(row["shots"]) != 10000:
            raise RuntimeError(f"Fold {fold} shot count changed")
        if float(row["margin_lcb"]) <= 0:
            raise RuntimeError(f"Fold {fold} margin is not positive")
        if row["pass"].lower() != "true":
            raise RuntimeError(f"Fold {fold} is not marked as passing")

    for fold, alias in EXPECTED_ALIASES.items():
        qasm = PUBLIC / "qasm" / f"{alias}.qasm3"
        if not qasm.is_file():
            raise RuntimeError(f"Missing public QASM: {qasm.name}")
        counts = count_qasm(qasm)
        expected = {
            "rx": EXPECTED_RX[fold],
            "rz": EXPECTED_RZ[fold],
            "cz": EXPECTED_CZ[fold],
            "barrier": EXPECTED_BARRIERS[fold],
            "measure": EXPECTED_MEASUREMENTS[fold],
        }
        if counts != expected:
            raise RuntimeError(
                f"QASM gate-count mismatch for {alias}: expected {expected}, got {counts}"
            )

    verify_manifest()
    print("CODEGAPQA PUBLIC RELEASE VERIFICATION: PASS")
    print("Public circuits: confirmatory-fold-1, confirmatory-fold-3, confirmatory-fold-5")
    print(f"Base margin LCB: {summary['base_margin_lcb']}")
    print(f"Minimum margin LCB: {summary['minimum_margin_lcb']}")
    print(f"Conditional gap: {summary['gamma_log10']} >= {summary['gamma_target']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CODEGAPQA PUBLIC RELEASE VERIFICATION: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
