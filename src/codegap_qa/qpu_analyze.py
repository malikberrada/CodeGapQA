from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import numpy as np

from .features import FeatureMap
from .qpu_counts import choose_count_convention, counts_to_samples
from .qpu_types import ProbeAnalysis
from .witness import MinimaxWitness, certificate_margin_lcb


def _result_map(payload: dict) -> dict[str, dict]:
    return {str(item['label']): item for item in payload['results']}


def analyze_probes(
    manifest: dict,
    raw_counts: dict,
    *,
    science_depth: int,
    minimum_convention_accuracy: float,
    maximum_readout_error: float,
    maximum_model_mismatch: float,
) -> ProbeAnalysis:
    results = _result_map(raw_counts)
    basis = [item for item in manifest['circuits'] if item['type'] == 'basis']
    expected = {
        item['label']: tuple(int(bit) for bit in item['expected_pattern'])
        for item in basis
    }
    labeled_counts = {label: results[label]['counts'] for label in expected}
    convention, convention_scores = choose_count_convention(labeled_counts, expected)
    n = int(manifest['logical_qubits'])
    all0 = counts_to_samples(results['basis_all0']['counts'], n, convention)
    all1 = counts_to_samples(results['basis_all1']['counts'], n, convention)
    p01 = all0.mean(axis=0)
    p10 = 1.0 - all1.mean(axis=0)
    readout_worst = float(max(p01.max(initial=0.0), p10.max(initial=0.0)))
    checker_errors = []
    for label in ('basis_even', 'basis_odd'):
        samples = counts_to_samples(results[label]['counts'], n, convention)
        target = np.asarray(expected[label], dtype=np.uint8)
        checker_errors.append(float(np.mean(samples != target[None, :])))
    checkerboard_error = float(max(checker_errors))
    echo_survival: dict[str, float] = {}
    xs, ys = [], []
    for item in manifest['circuits']:
        if item['type'] != 'echo':
            continue
        label = item['label']
        samples = counts_to_samples(results[label]['counts'], n, convention)
        survival = float(np.mean(np.all(samples == 0, axis=1)))
        echo_survival[label] = survival
        if survival > 0:
            xs.append(float(item['repetitions']))
            ys.append(math.log(survival))
    if len(xs) >= 2:
        x_values = np.asarray(xs, dtype=np.float64)
        y_values = np.asarray(ys, dtype=np.float64)
        # Each echo repetition applies U followed by U^-1. Fit
        # log F_echo(r) = -2 r L through the physical origin F(0)=1.
        slope = float(np.dot(x_values, y_values) / np.dot(x_values, x_values))
        science_survival = float(np.clip(math.exp(slope / 2.0), 0.0, 1.0))
        fitted = np.exp(slope * x_values)
        model_mismatch = float(np.max(np.abs(fitted - np.exp(y_values))))
    else:
        science_survival = math.sqrt(max(echo_survival.get('echo_x1', 0.0), 0.0))
        model_mismatch = 1.0
    # Conservative proxy: a distribution with ideal-outcome survival F can be
    # at total-variation distance as large as 1-F.
    tv_proxy = 1.0 - science_survival
    reasons = []
    if convention_scores[convention] < minimum_convention_accuracy:
        reasons.append('bit_order_probe_failed')
    if readout_worst > maximum_readout_error:
        reasons.append('readout_error_above_limit')
    if model_mismatch > maximum_model_mismatch:
        reasons.append('echo_model_mismatch')
    return ProbeAnalysis(
        candidate_id=str(manifest.get('candidate_id', 'unknown')),
        convention=convention,
        convention_accuracy=float(convention_scores[convention]),
        readout_p01_mean=float(p01.mean()),
        readout_p10_mean=float(p10.mean()),
        readout_worst=readout_worst,
        checkerboard_error=checkerboard_error,
        echo_survival=echo_survival,
        science_survival_proxy=science_survival,
        tv_noise_proxy=tv_proxy,
        model_mismatch=model_mismatch,
        passed_integrity=not reasons,
        reasons=tuple(reasons),
    )


def load_witness(path: Path) -> tuple[FeatureMap, MinimaxWitness, float]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    fmap_payload = payload['feature_map']
    n = int(fmap_payload['n'])
    parity_masks = np.asarray(fmap_payload['parity_masks'], dtype=np.uint8)
    if parity_masks.size == 0:
        parity_masks = np.empty((0, n), dtype=np.uint8)
    elif parity_masks.ndim != 2 or parity_masks.shape[1] != n:
        raise ValueError('Stored parity masks have the wrong shape.')
    # CODEGAP_V075_WITNESS_FEATURE_MAP_RESTORE
    # Reconstruct the exact preregistered lightcone feature subset.
    # The former loader ignored bit_indices/include_centered_weight,
    # turning a 16-dimensional witness into 65 runtime features for n=48.
    stored_bit_indices = fmap_payload.get('bit_indices')
    bit_indices = (
        None
        if stored_bit_indices is None
        else tuple(int(value) for value in stored_bit_indices)
    )
    include_centered_weight = bool(
        fmap_payload.get('include_centered_weight', True)
    )
    fmap = FeatureMap(
        parity_masks=parity_masks,
        heavy_indices=tuple(
            int(value) for value in fmap_payload['heavy_indices']
        ),
        n=n,
        bit_indices=bit_indices,
        include_centered_weight=include_centered_weight,
    )
    witness_payload = payload['witness']
    weights = np.asarray(
        witness_payload['weights'],
        dtype=np.float64,
    )
    if weights.ndim != 1:
        raise ValueError(
            'Stored witness weights must be one-dimensional.'
        )
    feature_names = tuple(
        witness_payload['feature_names']
    )
    if fmap.dimension != int(weights.size):
        raise ValueError(
            'Stored feature-map dimension does not match '
            'witness weights: '
            f'{fmap.dimension} != {weights.size}.'
        )
    if len(feature_names) != int(weights.size):
        raise ValueError(
            'Stored feature-name count does not match '
            'witness weights: '
            f'{len(feature_names)} != {weights.size}.'
        )
    witness = MinimaxWitness(
        weights=weights,
        feature_names=feature_names,
        training_margin=float(
            witness_payload['training_margin']
        ),
        adversary_means={
            str(key): float(value)
            for key, value
            in witness_payload['adversary_means'].items()
        },
        ideal_mean=float(
            witness_payload['ideal_mean']
        ),
    )
    penalty = float(
        payload.get(
            'adversary_generalization_penalty',
            0.0,
        )
    )
    return fmap, witness, penalty


def certify_qpu_counts(
    *,
    raw_counts: dict,
    label: str,
    convention: str,
    witness_path: Path,
    alpha: float,
    adversary_generalization_penalty: float,
) -> dict:
    results = _result_map(raw_counts)
    fmap, witness, stored_penalty = load_witness(witness_path)
    samples = counts_to_samples(results[label]['counts'], fmap.n, convention)
    penalty = max(float(adversary_generalization_penalty), stored_penalty)
    certificate = certificate_margin_lcb(
        witness,
        fmap.transform(samples),
        alpha=alpha,
        adversary_generalization_penalty=penalty,
    )
    certificate.update(
        {
            'schema': 'codegap.qpu-minimax-certificate.v1',
            'label': label,
            'convention': convention,
            'raw_counts_only': True,
            'readout_mitigation_used': False,
            'acceptance_rule': 'margin_lcb > 0',
        }
    )
    return certificate
