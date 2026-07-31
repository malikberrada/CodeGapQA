import json
from pathlib import Path

from codegap_qa.freeze import freeze
from codegap_qa.qpu_workflow import certify_science, select_from_probes


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def test_offline_probe_selection_and_diagnostic(tmp_path: Path):
    source = tmp_path / 'source'
    campaign = tmp_path / 'campaign'
    probe_run = tmp_path / 'probes'
    selected = tmp_path / 'selected'
    diagnostic_run = tmp_path / 'diagnostic'
    certificate_root = tmp_path / 'certificate'
    candidate_id = 'candidate-1'
    candidate_root = campaign / 'candidates' / candidate_id

    config = {
        'qpu': {
            'backend': 'mock:qpu',
            'probe_shots': 100,
            'diagnostic_shots': 200,
            'final_shots': 500,
            'probe_acceptance': {
                'minimum_convention_accuracy': 0.99,
                'maximum_readout_error': 0.01,
                'maximum_model_mismatch': 0.01,
                'tv_radius_safety_factor': 0.5,
            },
            'diagnostic': {
                'alpha': 0.05,
                'adversary_generalization_penalty': 0.01,
                'minimum_margin_lcb': 0.0,
            },
            'final': {
                'alpha': 0.01,
                'adversary_generalization_penalty': 0.01,
                'minimum_margin_lcb': 0.0,
            },
        }
    }
    _write(source / 'resolved_config.json', config)

    manifest = {
        'candidate_id': candidate_id,
        'logical_qubits': 4,
        'layout': [0, 1, 2, 3],
        'circuits': [
            {'label': 'basis_all0', 'type': 'basis', 'expected_pattern': [0, 0, 0, 0]},
            {'label': 'basis_all1', 'type': 'basis', 'expected_pattern': [1, 1, 1, 1]},
            {'label': 'basis_even', 'type': 'basis', 'expected_pattern': [0, 1, 0, 1]},
            {'label': 'basis_odd', 'type': 'basis', 'expected_pattern': [1, 0, 1, 0]},
            {'label': 'echo_x1', 'type': 'echo', 'repetitions': 1, 'expected_pattern': [0, 0, 0, 0]},
            {'label': 'echo_x2', 'type': 'echo', 'repetitions': 2, 'expected_pattern': [0, 0, 0, 0]},
            {'label': 'echo_x4', 'type': 'echo', 'repetitions': 4, 'expected_pattern': [0, 0, 0, 0]},
        ],
    }
    _write(candidate_root / 'probes' / 'probe_manifest.json', manifest)
    (candidate_root / 'science.qasm3').write_text('OPENQASM 3.0;\n', encoding='utf-8')
    witness = {
        'feature_map': {'parity_masks': [], 'heavy_indices': [0], 'n': 4},
        'witness': {
            'weights': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            'feature_names': ['b0', 'b1', 'b2', 'b3', 'heavy0', 'weight'],
            'training_margin': 1.5,
            'adversary_means': {'uniform': -0.5},
            'ideal_mean': 1.0,
        },
        'adversary_generalization_penalty': 0.01,
    }
    _write(candidate_root / 'robust_witness.json', witness)
    prepared = {
        'candidate': {
            'candidate_id': candidate_id,
            'code': {'n': 4},
            'objective': 10.0,
            'hardness': {'gamma_log10': 3.0},
        },
        'compile': {
            'best': {
                'two_qubit_depth': 2,
                'two_qubit_count': 4,
                'swap_count': 0,
                'calibrated_log_error': 0.01,
            }
        },
        'offline_noise': {'tv_robust_radius': 0.4},
        'gates_abc': [
            {'passed': True},
            {'passed': True},
            {'passed': True, 'claim_level': 'CONDITIONAL_HARDNESS'},
        ],
        'witness_path': str(candidate_root / 'robust_witness.json'),
        'science_qasm': str(candidate_root / 'science.qasm3'),
        'probe_manifest': str(candidate_root / 'probes' / 'probe_manifest.json'),
        'ready_for_qpu_probes': True,
    }
    _write(candidate_root / 'prepared_candidate.json', prepared)
    _write(
        campaign / 'campaign_manifest.json',
        {
            'backend': 'mock:qpu',
            'source_artifact': str(source),
            'candidates': [
                {
                    'candidate_id': candidate_id,
                    'ready_for_qpu_probes': True,
                }
            ],
        },
    )
    freeze(campaign)

    clean = {
        'basis_all0': '0000',
        'basis_all1': '1111',
        'basis_even': '1010',
        'basis_odd': '0101',
        'echo_x1': '0000',
        'echo_x2': '0000',
        'echo_x4': '0000',
    }
    results = [
        {
            'label': f'{candidate_id}::{label}',
            'shots_received': 100,
            'counts': {key: 100},
        }
        for label, key in clean.items()
    ]
    _write(
        probe_run / 'raw_counts.json',
        {'backend': 'mock:qpu', 'stage': 'probes', 'results': results},
    )
    freeze(probe_run)

    selection = select_from_probes(
        campaign_root=campaign,
        probe_run=probe_run,
        output=selected,
    )
    assert selection['status'] == 'PASS'
    assert selection['selected']['candidate_id'] == candidate_id

    _write(
        diagnostic_run / 'raw_counts.json',
        {
            'backend': 'mock:qpu',
            'stage': 'diagnostic',
            'results': [
                {
                    'label': 'science',
                    'shots_received': 200,
                    'counts': {'0000': 200},
                }
            ],
        },
    )
    freeze(diagnostic_run)
    certificate = certify_science(
        selected_root=selected,
        science_run=diagnostic_run,
        stage='diagnostic',
        output=certificate_root,
    )
    assert certificate['pass']
    gates = json.loads((selected / 'qpu_gate_report.json').read_text())
    assert gates['D_QPU_DIAGNOSTIC']['passed']
