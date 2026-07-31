# CodeGapQA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21724982.svg)](https://doi.org/10.5281/zenodo.21724982)

CodeGapQA is a research framework for hardware-oriented quantum-circuit
co-design, local-witness certification, finite-shot uncertainty analysis, and
registered classical-attack benchmarking.

This public repository accompanies the preprint:

> **Hardware-Oriented Hidden-Code-Sampling-Inspired Certification on a
> 60-Qubit Superconducting Processor**

The reported experiment is HCS-inspired. It is not an end-to-end implementation
of the original Hidden Code Sampling logical/syndrome verification protocol.

## Public certificate

The sanitized public result is:

| Quantity | Value |
|---|---:|
| Status | `QPU_PANEL_PASS` |
| Active qubits | 60 |
| Public circuits | `confirmatory-fold-1`, `confirmatory-fold-3`, `confirmatory-fold-5` |
| Science shots | 10,000 per fold |
| Base-fold margin LCB | 0.2952588808 |
| Minimum margin LCB | 0.2857977493 |
| Folds passed | 3/3 |
| Conditional gap | 6.5553534694 |
| Registered gap target | 6.0 |

The conditional gap is an operation-count comparison against the registered
classical attack suite. It is not a wall-clock quantum-advantage claim and does
not exclude future structure-specific classical algorithms.

## Repository layout

```text
src/codegap_qa/          Python research package
native/                  Optional C++ acceleration source
configs/                 Current public configurations
scripts/                 Scientific preparation, analysis, and verification tools
tests/                   Unit and regression tests
public_release/qasm/     Sanitized QASM with public fold names
public_release/data/     Aggregate certificate metrics
public_release/paper/    Preprint PDF and LaTeX source
docs/                    Reproducibility, scope, and attribution notes
```

The repository intentionally excludes provider credentials, real provider job
identifiers, account or organization metadata, private submission ledgers,
local backups, and operational attempt history.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,tensor]"
```

Optional QPU client dependencies:

```bash
python -m pip install -e ".[qpu]"
```

No credentials are required for public-data verification.

## Verify the public release

```bash
python scripts/verify_public_release.py
```

This checks the manifest, the three public aliases, QASM hashes and gate counts,
positive certified margins, the three-fold pass, and the registered gap target.

## Run core tests

```bash
pytest -q \
  tests/test_gf2.py \
  tests/test_bicycle.py \
  tests/test_witness.py \
  tests/test_lightcone_v040.py \
  tests/test_v165_lightcone_endianness.py \
  tests/test_freeze.py
```

## Public QASM names

Only these public names are used for the confirmatory circuits:

```text
confirmatory-fold-1
confirmatory-fold-3
confirmatory-fold-5
```

They correspond to local unitary-fold factors 1, 3, and 5. The public copies
retain the scientific circuit content and omit provider and account records.

## Hardware attribution

Quantum circuits were submitted to the Rigetti Cepheus-1-108Q superconducting
processor through the Open Quantum platform, operated by Quantum Rings, using
the Open Quantum Python SDK.

This acknowledgment does not imply endorsement by Quantum Rings or Rigetti
Computing of the findings, interpretations, or conclusions.

## Licences

- Source code: Apache License 2.0 (`LICENSE`).
- Paper, figures, public data, and public QASM: CC BY 4.0
  (`LICENSE-DATA.md`).

## Citation

Berrada, A. (2026). Hardware-Oriented Hidden-Code-Sampling-Inspired Certification on a 60-Qubit Superconducting Processor (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21724982

The archived preprint is permanently available at [10.5281/zenodo.21724982](https://doi.org/10.5281/zenodo.21724982).
Machine-readable citation metadata are provided in `CITATION.cff`.
