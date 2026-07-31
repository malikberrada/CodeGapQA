# Reproducibility

## Public verification

The public verification path is offline and requires no QPU credentials:

```bash
python -m pip install -e ".[dev,tensor]"
python scripts/verify_public_release.py
```

The script verifies:

1. the public aliases `confirmatory-fold-1`, `confirmatory-fold-3`, and
   `confirmatory-fold-5`;
2. SHA-256 integrity of the public files;
3. the registered QASM gate counts at folds 1, 3, and 5;
4. 10,000 science shots per fold;
5. positive robust margins and three passing folds;
6. the conditional gap against its registered target.

## Scientific pipeline

The public source includes the reusable algorithms for code/topology search,
light-cone witness evaluation, finite-shot analysis, and registered classical
attack estimates. The release-specific offline verifier checks the sanitized
QASM and aggregate certificate metrics. Provider-specific execution ledgers and
operational recovery history are intentionally not published.

A new hardware campaign requires independent provider access and should create a
new versioned artifact chain. It must not overwrite the public v1 certificate.
