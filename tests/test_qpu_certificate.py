import json
from pathlib import Path

from codegap_qa.qpu_analyze import certify_qpu_counts


def test_qpu_certificate_positive(tmp_path: Path):
    witness = {
        'feature_map': {
            'parity_masks': [],
            'heavy_indices': [0],
            'n': 2,
        },
        'witness': {
            'weights': [0.0, 0.0, 1.0, 0.0],
            'feature_names': ['b0','b1','heavy0','weight'],
            'training_margin': 1.5,
            'adversary_means': {'uniform': -0.5},
            'ideal_mean': 1.0,
        },
        'adversary_generalization_penalty': 0.01,
    }
    path = tmp_path / 'witness.json'
    path.write_text(json.dumps(witness), encoding='utf-8')
    raw = {
        'results': [
            {'label':'science','shots_received':2000,'counts':{'00':2000}}
        ]
    }
    result = certify_qpu_counts(
        raw_counts=raw,
        label='science',
        convention='qiskit',
        witness_path=path,
        alpha=0.05,
        adversary_generalization_penalty=0.01,
    )
    assert result['pass']
    assert result['margin_lcb'] > 0
