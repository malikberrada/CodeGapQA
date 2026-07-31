from __future__ import annotations

import os

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

from .freeze import freeze, verify_freeze
from .gates import gate_a, gate_a_qpu, gate_b, gate_c
from .phase import evaluate_noise_phase
from .qpu_analyze import analyze_probes, certify_qpu_counts
from .qpu_local import analyze_local_probes
from .qpu_compile import compile_candidate
from .qpu_probes import build_probe_bundle
from .qpu_provider import OpenQuantumProvider
from .qpu_snapshot import (
    snapshot_target,
    structural_fingerprint,
    write_snapshot,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _order_candidates_for_preparation(
    frontier: list[dict[str, Any]],
    gate_report: dict[str, Any],
    gamma_threshold: float,
) -> list[dict[str, Any]]:
    selected_id = gate_report.get("selected_candidate_id")
    backup_id = gate_report.get("backup_candidate_id")
    priority: dict[str, int] = {}
    if selected_id:
        priority[str(selected_id)] = 0
    if backup_id and str(backup_id) not in priority:
        priority[str(backup_id)] = 1
    default_priority = len(priority) + 1
    candidates = [
        item
        for item in frontier
        if float(item["hardness"]["gamma_log10"]) >= gamma_threshold
    ]
    return sorted(
        candidates,
        key=lambda item: (
            priority.get(str(item.get("candidate_id")), default_priority),
            int(item["code"]["n"]),
            -float(item.get("objective", 0.0)),
        ),
    )

def capture_qpu_target_snapshot(
    *,
    credentials: Path | None,
    backend_name: str,
    output: Path,
    execution_plan: str,
    queue_priority: str,
) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with OpenQuantumProvider(credentials) as provider:
        metadata = provider.backend_metadata(backend_name)
        backend = provider.backend(backend_name, execution_plan, queue_priority)
        snapshot = snapshot_target(
            backend,
            backend_name=backend_name,
            accepting_jobs=metadata.get("accepting_jobs"),
            queue_depth=metadata.get("queue_depth"),
        )
        payload = write_snapshot(snapshot, output / "target_snapshot.json")
    return {
        "schema": "codegap.target-snapshot-command.v1",
        "status": "PASS",
        "backend": backend_name,
        "target_snapshot": str(output / "target_snapshot.json"),
        "structural_fingerprint": payload["structural_fingerprint"],
        "calibration_fingerprint": payload["calibration_fingerprint"],
        "num_qubits": payload["num_qubits"],
        "coupling_edges": len(payload["coupling_edges"]),
    }

def prepare_qpu_campaign(
    *,
    artifact_root: Path,
    credentials: Path | None,
    backend_name: str,
    output: Path,
    max_candidates: int,
    max_layouts: int,
    execution_plan: str,
    queue_priority: str,
) -> dict:
    artifact_root = artifact_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_freeze = artifact_root / 'freeze_manifest.json'
    if not source_freeze.is_file() or not verify_freeze(source_freeze)['ok']:
        raise RuntimeError('The classical research artifact must be frozen before QPU preparation.')
    config = _load(artifact_root / 'resolved_config.json')
    local_protocol: dict[str, Any] | None = None
    raw_protocol_path = os.environ.get(
        "CODEGAP_LOCAL_PROTOCOL_CONFIG",
        "",
    ).strip()
    if raw_protocol_path:
        protocol_path = Path(raw_protocol_path).resolve()
        if not protocol_path.is_file():
            raise RuntimeError(
                "CODEGAP_LOCAL_PROTOCOL_CONFIG does not point to a file."
            )
        local_protocol = _load(protocol_path)
        if local_protocol.get("mode") != "local_witness_v1":
            raise RuntimeError(
                "Unsupported local QPU protocol mode."
            )
        (output / "local_protocol.json").write_text(
            json.dumps(local_protocol, indent=2) + "\n",
            encoding="utf-8",
        )
    gate_report = _load(artifact_root / 'gate_report.json')
    if not gate_report.get('qpu_preparation_authorized', False):
        raise RuntimeError(
            'QPU preparation is blocked: no candidate passed static code, '
            'NoiseCert and Gate C after adversarial schedule search.'
        )
    frontier = _load(artifact_root / 'codeforge' / 'hardware_code_frontier.json')
    optimality = _load(artifact_root / 'codeforge' / 'optimality_certificate.json')
    gamma_threshold = float(config['gates']['min_gamma_log10'])
    candidates = _order_candidates_for_preparation(
        frontier, gate_report, gamma_threshold
    )[:max_candidates]
    if not candidates:
        raise RuntimeError('No Gate-C-passing candidate is available for QPU preparation.')

    # CODEGAP_V068_LAYOUT_BLACKLIST
    raw_excluded = os.environ.get(
        "CODEGAP_EXCLUDED_PHYSICAL_QUBITS",
        "",
    )

    tokens = [
        token.strip()
        for token in raw_excluded.replace(
            ";",
            ",",
        ).split(",")
        if token.strip()
    ]

    try:
        excluded_physical_qubits = tuple(
            sorted(
                {
                    int(token)
                    for token in tokens
                }
            )
        )
    except ValueError as error:
        raise RuntimeError(
            "CODEGAP_EXCLUDED_PHYSICAL_QUBITS must "
            "contain comma-separated integers."
        ) from error

    if any(
        value < 0
        for value in excluded_physical_qubits
    ):
        raise RuntimeError(
            "Excluded physical qubits must be non-negative."
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    layout_constraints_payload = {
        "schema": "codegap.qpu-layout-constraints.v1",
        "source": "CODEGAP_EXCLUDED_PHYSICAL_QUBITS",
        "excluded_physical_qubits": list(
            excluded_physical_qubits
        ),
        "policy": (
            "Remove excluded nodes from the live target "
            "graph and enumerate a fresh zero-SWAP layout "
            "when the pinned layout uses an excluded node."
        ),
    }

    (
        output / "layout_constraints.json"
    ).write_text(
        json.dumps(
            layout_constraints_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with OpenQuantumProvider(credentials) as provider:
        metadata = provider.backend_metadata(backend_name)
        if metadata.get('accepting_jobs') is False:
            raise RuntimeError(f'Backend {backend_name!r} is not accepting jobs.')
        backend = provider.backend(backend_name, execution_plan, queue_priority)
        snapshot = snapshot_target(
            backend,
            backend_name=backend_name,
            accepting_jobs=metadata.get('accepting_jobs'),
            queue_depth=metadata.get('queue_depth'),
        )
        snapshot_payload = write_snapshot(snapshot, output / 'target_snapshot.json')
        configured_snapshot = config.get('hardware', {}).get('target_snapshot')
        if configured_snapshot:
            configured_path = Path(configured_snapshot)
            if not configured_path.is_absolute():
                configured_path = Path(config.get('_config_dir', artifact_root)) / configured_path
            preregistered = _load(configured_path.resolve())
            expected_fingerprint = preregistered.get('structural_fingerprint')
            actual_fingerprint = snapshot_payload.get('structural_fingerprint')
            if expected_fingerprint and expected_fingerprint != actual_fingerprint:
                raise RuntimeError(
                    'Cepheus target structure changed since schedule search. '
                    'Capture a new snapshot and rerun the classical pipeline.'
                )
        prepared = []
        for index, candidate in enumerate(candidates):
            candidate_root = output / 'candidates' / candidate['candidate_id']
            candidate_root.mkdir(parents=True, exist_ok=True)
            candidate = dict(candidate)
            compile_report = compile_candidate(
                candidate,
                backend,
                candidate_root / 'layouts',
                max_layouts=max_layouts,
                seed=config['seed'] + index * 100003,
                excluded_physical_qubits=(
                    excluded_physical_qubits
                ),
            )

            best_layout = tuple(
                int(value)
                for value in (
                    compile_report
                    .get("best", {})
                    .get("layout", [])
                )
            )

            excluded_used = sorted(
                set(best_layout).intersection(
                    excluded_physical_qubits
                )
            )

            excluded_respected = (
                not bool(excluded_used)
                if best_layout
                else None
            )

            compile_report["layout_constraints"] = {
                "excluded_physical_qubits": list(
                    excluded_physical_qubits
                ),
                "excluded_physical_qubits_used": (
                    excluded_used
                ),
                "excluded_qubits_respected": (
                    excluded_respected
                ),
            }

            if (
                compile_report.get("status") == "PASS"
                and excluded_respected is not True
            ):
                raise RuntimeError(
                    "A PASS compilation uses excluded "
                    f"physical qubits: {excluded_used}"
                )

            (candidate_root / 'compile_report.json').write_text(
                json.dumps(compile_report, indent=2), encoding='utf-8'
            )
            static_a = gate_a(candidate, optimality, config)
            qpu_a = gate_a_qpu(candidate, compile_report, config)
            if compile_report['status'] != 'PASS':
                prepared.append(
                    {
                        'candidate_id': candidate['candidate_id'],
                        'status': compile_report['status'],
                        'n': candidate['code']['n'],
                        'ready_for_qpu_probes': False,
                        'A_CODE_QPU': asdict(qpu_a),
                    }
                )
                continue
            selected_qasm = candidate_root / 'science.qasm3'
            logical_qasm = candidate_root / 'logical_science.qasm3'
            shutil.copy2(compile_report['best']['qasm_path'], selected_qasm)
            shutil.copy2(compile_report['logical_qasm_path'], logical_qasm)
            compile_report['best']['qasm_path'] = str(selected_qasm)
            noise_root = candidate_root / 'offline_noisecert'
            noise_result = evaluate_noise_phase(candidate, config, noise_root)
            witness_source = noise_root / 'robust_witness.json'
            witness_payload = _load(witness_source)
            witness_payload['adversary_generalization_penalty'] = config['certificate']['adversary_generalization_penalty']
            witness_path = candidate_root / 'robust_witness.json'
            witness_path.write_text(
                json.dumps(witness_payload, indent=2), encoding='utf-8'
            )
            b = gate_b(noise_result, config)
            c = gate_c(candidate, config)
            probe_manifest = build_probe_bundle(
                science_qasm=logical_qasm,
                backend=backend,
                layout=tuple(compile_report['best']['layout']),
                output=candidate_root / 'probes',
                seed=config['seed'] + index * 100003 + 5000,
                candidate=candidate,
                protocol_config=local_protocol,
            )
            probe_manifest['candidate_id'] = candidate['candidate_id']
            (candidate_root / 'probes' / 'probe_manifest.json').write_text(
                json.dumps(probe_manifest, indent=2), encoding='utf-8'
            )
            local_science_manifest = (
                candidate_root / 'probes' / 'local_science_manifest.json'
            )
            gate_payload = {
                'A_CODE_STATIC': asdict(static_a),
                'A_CODE_QPU': asdict(qpu_a),
                'B_NOISE': asdict(b),
                'C_HARDNESS': asdict(c),
            }
            ready = bool(qpu_a.passed and b.passed and c.passed)
            candidate_payload = {
                'candidate': candidate,
                'compile': compile_report,
                'offline_noise': noise_result,
                'gates': gate_payload,
                'gates_abc': [asdict(static_a), asdict(b), asdict(c)],
                'witness_path': str(witness_path),
                'science_qasm': str(selected_qasm),
                'logical_science_qasm': str(logical_qasm),
                'probe_manifest': str(candidate_root / 'probes' / 'probe_manifest.json'),
                'local_science_manifest': (
                    str(local_science_manifest)
                    if local_science_manifest.is_file()
                    else None
                ),
                'qpu_protocol': probe_manifest.get('protocol', 'legacy_global_v1'),
                'ready_for_qpu_probes': ready,
            }
            (candidate_root / 'prepared_candidate.json').write_text(
                json.dumps(candidate_payload, indent=2), encoding='utf-8'
            )
            prepared.append(
                {
                    'candidate_id': candidate['candidate_id'],
                    'status': 'PASS' if ready else 'STOP_GATES_A_B_C',
                    'n': candidate['code']['n'],
                    'ready_for_qpu_probes': ready,
                    'root': str(candidate_root),
                    'compile_score': compile_report['best']['score'],
                    'A_CODE_QPU': asdict(qpu_a),
                    'B_NOISE': asdict(b),
                    'C_HARDNESS': asdict(c),
                    'qpu_protocol': probe_manifest.get(
                        'protocol',
                        'legacy_global_v1',
                    ),
                }
            )
    manifest = {
        'schema': 'codegap.qpu-campaign.v2',
        'created_at': utc_now(),
        'backend': backend_name,
        'execution_plan': execution_plan,
        'queue_priority': queue_priority,
        'source_artifact': str(artifact_root),
        'source_freeze_verified': True,
        'target_snapshot': str(output / 'target_snapshot.json'),
        'target_snapshot_sha256': snapshot_payload['sha256'],
        'local_protocol': (
            str(output / 'local_protocol.json')
            if local_protocol is not None
            else None
        ),
        'local_protocol_sha256': (
            _digest(output / 'local_protocol.json')
            if local_protocol is not None
            else None
        ),
        'candidates': prepared,
    }
    (output / 'campaign_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    freeze(output)
    return manifest


def _candidate_paths(campaign_root: Path, candidate_id: str) -> tuple[dict, Path]:
    root = campaign_root / 'candidates' / candidate_id
    return _load(root / 'prepared_candidate.json'), root



def _probe_submission_plan(campaign_root: Path) -> tuple[dict, list[Path], list[str]]:
    campaign = _load(campaign_root / 'campaign_manifest.json')
    qasm_files: list[Path] = []
    labels: list[str] = []
    for item in campaign['candidates']:
        if not item.get('ready_for_qpu_probes'):
            continue
        prepared, _ = _candidate_paths(campaign_root, item['candidate_id'])
        manifest = _load(Path(prepared['probe_manifest']))
        for circuit in manifest['circuits']:
            qasm_files.append(Path(circuit['qasm_path']))
            labels.append(f"{item['candidate_id']}::{circuit['label']}")
    if not qasm_files:
        raise RuntimeError('No candidate passed live Gate A QPU, offline Gate B, and Gate C.')
    return campaign, qasm_files, labels

def submit_probes(
    *, campaign_root: Path, credentials: Path | None, shots: int, output: Path
) -> dict:
    campaign, qasm_files, labels = _probe_submission_plan(campaign_root)
    if not verify_freeze(campaign_root / 'freeze_manifest.json')['ok']:
        raise RuntimeError('Campaign freeze verification failed.')
    with OpenQuantumProvider(credentials) as provider:
        result = provider.submit_bundle(
            backend_name=campaign['backend'],
            qasm_files=qasm_files,
            labels=labels,
            shots=shots,
            output=output,
            stage='probes',
            execution_plan=campaign['execution_plan'],
            queue_priority=campaign['queue_priority'],
        )
    freeze(output)
    return result


def attach_probe_job(
    *,
    campaign_root: Path,
    credentials: Path | None,
    shots: int,
    output: Path,
    job_id: str,
    label: str | None,
) -> dict:
    campaign, qasm_files, labels = _probe_submission_plan(campaign_root)
    if not verify_freeze(campaign_root / 'freeze_manifest.json')['ok']:
        raise RuntimeError('Campaign freeze verification failed.')
    with OpenQuantumProvider(credentials) as provider:
        return provider.attach_async_job(
            backend_name=campaign['backend'],
            qasm_files=qasm_files,
            labels=labels,
            shots=shots,
            output=output,
            stage='probes',
            execution_plan=campaign['execution_plan'],
            queue_priority=campaign['queue_priority'],
            job_id=job_id,
            label=label,
        )


def submit_probes_async(
    *,
    campaign_root: Path,
    credentials: Path | None,
    shots: int,
    output: Path,
    max_active: int,
    slot_poll_seconds: int,
    slot_timeout_seconds: int,
) -> dict:
    campaign, qasm_files, labels = _probe_submission_plan(campaign_root)
    if not verify_freeze(campaign_root / 'freeze_manifest.json')['ok']:
        raise RuntimeError('Campaign freeze verification failed.')
    with OpenQuantumProvider(credentials) as provider:
        return provider.submit_bundle_async(
            backend_name=campaign['backend'],
            qasm_files=qasm_files,
            labels=labels,
            shots=shots,
            output=output,
            stage='probes',
            execution_plan=campaign['execution_plan'],
            queue_priority=campaign['queue_priority'],
            max_active=max_active,
            slot_poll_seconds=slot_poll_seconds,
            slot_timeout_seconds=slot_timeout_seconds,
        )


def collect_probes_async(
    *,
    campaign_root: Path,
    credentials: Path | None,
    output: Path,
    wait: bool,
    poll_seconds: int,
    timeout_seconds: int,
    job_ids: list[str] | None = None,
) -> dict:
    if not verify_freeze(campaign_root / 'freeze_manifest.json')['ok']:
        raise RuntimeError('Campaign freeze verification failed.')
    with OpenQuantumProvider(credentials) as provider:
        result = provider.collect_async_bundle(
            output=output,
            wait=wait,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            job_ids=job_ids,
        )
    if result.get('status') == 'RESULTS_COLLECTED':
        freeze(output)
    return result


def import_probe_job(
    *,
    campaign_root: Path,
    credentials: Path | None,
    shots: int,
    output: Path,
    job_id: str,
    label: str,
) -> dict:
    campaign, qasm_files, labels = _probe_submission_plan(campaign_root)
    if label not in labels:
        raise RuntimeError(
            f'Unknown probe label {label!r}. Expected one of: {labels}.'
        )
    if not verify_freeze(campaign_root / 'freeze_manifest.json')['ok']:
        raise RuntimeError('Campaign freeze verification failed.')
    with OpenQuantumProvider(credentials) as provider:
        provider.submit_bundle(
            backend_name=campaign['backend'],
            qasm_files=qasm_files,
            labels=labels,
            shots=shots,
            output=output,
            stage='probes',
            execution_plan=campaign['execution_plan'],
            queue_priority=campaign['queue_priority'],
            prepare_only=True,
        )
        return provider.import_completed_job(
            output=output,
            job_id=job_id,
            label=label,
        )


def list_qpu_jobs(
    *, credentials: Path | None, status: str | None, limit: int
) -> dict:
    with OpenQuantumProvider(credentials) as provider:
        jobs = provider.list_jobs(status=status, limit=limit)
    return {
        'schema': 'codegap.openquantum-job-list.v1',
        'status_filter': status,
        'jobs': jobs,
    }


def select_from_probes(
    *, campaign_root: Path, probe_run: Path, output: Path
) -> dict:
    campaign_root = campaign_root.resolve()
    probe_run = probe_run.resolve()
    output = output.resolve()
    if not verify_freeze(campaign_root / 'freeze_manifest.json')['ok']:
        raise RuntimeError('Campaign freeze verification failed before probe analysis.')
    probe_freeze = probe_run / 'freeze_manifest.json'
    if not probe_freeze.is_file() or not verify_freeze(probe_freeze)['ok']:
        raise RuntimeError('Probe run must be frozen and verified before selection.')
    raw = _load(probe_run / 'raw_counts.json')
    campaign = _load(campaign_root / 'campaign_manifest.json')
    if raw.get('backend') != campaign['backend'] or raw.get('stage') != 'probes':
        raise RuntimeError('Probe result provenance does not match the campaign.')
    all_results = {str(item['label']): item for item in raw['results']}
    config = _load(Path(campaign['source_artifact']) / 'resolved_config.json')
    local_protocol_path = campaign_root / 'local_protocol.json'
    local_protocol = (
        _load(local_protocol_path)
        if local_protocol_path.is_file()
        else None
    )
    analyses = []
    for item in campaign['candidates']:
        if not item.get('ready_for_qpu_probes'):
            continue
        prepared, root = _candidate_paths(campaign_root, item['candidate_id'])
        manifest = _load(Path(prepared['probe_manifest']))
        candidate_results = []
        for probe in manifest['circuits']:
            compound = f"{item['candidate_id']}::{probe['label']}"
            copied = dict(all_results[compound])
            copied['label'] = probe['label']
            candidate_results.append(copied)
        raw_candidate = {'results': candidate_results}

        if manifest.get('protocol') == 'local_witness_v1':
            if local_protocol is None:
                raise RuntimeError(
                    'The local protocol manifest is missing from the campaign.'
                )
            analysis_payload = analyze_local_probes(
                manifest,
                raw_candidate,
                protocol_config=local_protocol,
            )
            noise_pass = bool(analysis_payload['qpu_noise_gate_pass'])
            reasons = list(analysis_payload['reasons'])
            analysis_payload.update(
                {
                    'tv_robust_radius': None,
                    'tv_radius_safety_factor': None,
                    'tv_noise_proxy': None,
                    'science_survival_proxy': None,
                    'model_mismatch': None,
                    'selection_metric': 'local_noise_score',
                }
            )
        else:
            qpu_limits = config['qpu']['probe_acceptance']
            analysis = analyze_probes(
                manifest,
                raw_candidate,
                science_depth=int(prepared['compile']['best']['two_qubit_depth']),
                minimum_convention_accuracy=qpu_limits['minimum_convention_accuracy'],
                maximum_readout_error=qpu_limits['maximum_readout_error'],
                maximum_model_mismatch=qpu_limits['maximum_model_mismatch'],
            )
            tv_radius = float(prepared['offline_noise']['tv_robust_radius'])
            safety = float(qpu_limits['tv_radius_safety_factor'])
            noise_pass = analysis.tv_noise_proxy < tv_radius * safety
            reasons = list(analysis.reasons)
            if not noise_pass:
                reasons.append('qpu_noise_proxy_exceeds_robust_radius')
            analysis_payload = analysis.to_dict()
            analysis_payload.update(
                {
                    'protocol': 'legacy_global_v1',
                    'tv_robust_radius': tv_radius,
                    'tv_radius_safety_factor': safety,
                    'qpu_noise_gate_pass': noise_pass,
                    'reasons': reasons,
                    'selection_metric': 'tv_noise_proxy',
                }
            )

        analysis_payload.update(
            {
                'candidate_root': str(root),
                'n': prepared['candidate']['code']['n'],
                'objective': prepared['candidate']['objective'],
                'calibrated_log_error': prepared['compile']['best']['calibrated_log_error'],
                'two_qubit_count': prepared['compile']['best']['two_qubit_count'],
                'two_qubit_depth': prepared['compile']['best']['two_qubit_depth'],
                'swap_count': prepared['compile']['best']['swap_count'],
                'gamma_log10': prepared['candidate']['hardness']['gamma_log10'],
                'A_CODE_QPU': prepared.get('gates', {}).get(
                    'A_CODE_QPU',
                    prepared.get('gates_abc', [{'passed': False}])[0],
                ),
                'B_NOISE_OFFLINE': prepared.get('gates', {}).get(
                    'B_NOISE',
                    prepared.get('gates_abc', [{}, {'passed': False}])[1],
                ),
                'C_HARDNESS_GAP': prepared.get('gates', {}).get(
                    'C_HARDNESS',
                    prepared.get('gates_abc', [{}, {}, {'passed': False}])[2],
                ),
            }
        )
        analyses.append(analysis_payload)

    eligible = [
        item
        for item in analyses
        if item['passed_integrity']
        and item['qpu_noise_gate_pass']
        and item['A_CODE_QPU'].get('passed', False)
        and item['C_HARDNESS_GAP'].get('passed', False)
    ]

    def selection_key(item: dict) -> tuple:
        if item.get('protocol') == 'local_witness_v1':
            maximum_lightcone = max(
                (
                    len(feature['lightcone'])
                    for feature in item.get('active_features', [])
                ),
                default=10**9,
            )
            return (
                maximum_lightcone,
                float(item.get('local_noise_score', 1.0)),
                item['n'],
                item['calibrated_log_error'],
                -item['objective'],
            )
        return (
            item['n'],
            float(item.get('tv_noise_proxy', 1.0)),
            item['calibrated_log_error'],
            -item['objective'],
        )

    eligible.sort(key=selection_key)
    selected = eligible[0] if eligible else None
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'schema': (
            'codegap.qpu-selection.v3'
            if local_protocol is not None
            else 'codegap.qpu-selection.v2'
        ),
        'created_at': utc_now(),
        'status': 'PASS' if selected else 'STOP_NO_QPU_ROBUST_CANDIDATE',
        'selection_rule': (
            local_protocol['selection_rule']
            if local_protocol is not None
            else (
                'minimum n passing bit-order/readout/echo integrity and the '
                'pre-registered TV robustness proxy; noise and calibration break ties'
            )
        ),
        'selected': selected,
        'analyses': analyses,
        'probe_run': str(probe_run),
        'probe_run_sha256': _digest(probe_run / 'raw_counts.json'),
        'campaign_root': str(campaign_root.resolve()),
        'qpu_config': {
            **config['qpu'],
            'local_protocol': local_protocol,
        },
    }
    (output / 'qpu_selection.json').write_text(
        json.dumps(report, indent=2) + '\n',
        encoding='utf-8',
    )

    qpu_pareto = []
    for candidate in analyses:
        candidate_noise = (
            float(candidate.get('local_noise_score', 1.0))
            if candidate.get('protocol') == 'local_witness_v1'
            else float(candidate.get('tv_noise_proxy', 1.0))
        )
        dominated = False
        for other in analyses:
            if other is candidate:
                continue
            other_noise = (
                float(other.get('local_noise_score', 1.0))
                if other.get('protocol') == 'local_witness_v1'
                else float(other.get('tv_noise_proxy', 1.0))
            )
            if (
                other['n'] <= candidate['n']
                and other['two_qubit_count'] <= candidate['two_qubit_count']
                and other['two_qubit_depth'] <= candidate['two_qubit_depth']
                and other_noise <= candidate_noise
                and other['gamma_log10'] >= candidate['gamma_log10']
                and (
                    other['n'] < candidate['n']
                    or other['two_qubit_count'] < candidate['two_qubit_count']
                    or other['two_qubit_depth'] < candidate['two_qubit_depth']
                    or other_noise < candidate_noise
                    or other['gamma_log10'] > candidate['gamma_log10']
                )
            ):
                dominated = True
                break
        if not dominated:
            qpu_pareto.append(candidate)
    (output / 'qpu_hardware_pareto.json').write_text(
        json.dumps(qpu_pareto, indent=2) + '\n',
        encoding='utf-8',
    )

    if selected:
        prepared, root = _candidate_paths(
            campaign_root,
            selected['candidate_id'],
        )
        witness_destination = output / 'robust_witness.json'
        shutil.copy2(prepared['witness_path'], witness_destination)
        (output / 'selected_candidate.json').write_text(
            json.dumps(prepared, indent=2) + '\n',
            encoding='utf-8',
        )
        shutil.copy2(prepared['science_qasm'], output / 'science.qasm3')
        if selected.get('protocol') == 'local_witness_v1':
            source_manifest = Path(prepared['local_science_manifest'])
            local_manifest = _load(source_manifest)
            # CODEGAP_V077_LOCAL_WITNESS_BINDING
            # The selected local protocol can use a hardware-aware witness
            # different from the candidate's original offline witness. Bind
            # the exact selected weights before any science execution.
            witness_payload = _load(witness_destination)
            selected_witness = dict(local_manifest['witness'])
            feature_map = witness_payload['feature_map']
            feature_dimension = (
                len(feature_map.get('bit_indices') or [])
                + len(feature_map.get('parity_masks') or [])
                + len(feature_map.get('heavy_indices') or [])
                + int(bool(feature_map.get('include_centered_weight', True)))
            )
            weights = selected_witness.get('weights') or []
            names = selected_witness.get('feature_names') or []
            if len(weights) != feature_dimension:
                raise RuntimeError(
                    'Selected local witness dimension does not match the '
                    f'stored feature map: {len(weights)} != {feature_dimension}.'
                )
            if len(names) != len(weights):
                raise RuntimeError(
                    'Selected local witness names do not match its weights.'
                )
            witness_payload['witness'] = selected_witness
            witness_payload['selection_binding'] = {
                'schema': 'codegap.local-witness-binding.v1',
                'source': str(source_manifest),
                'protocol': selected.get('protocol'),
                'candidate_id': selected.get('candidate_id'),
            }
            witness_destination.write_text(
                json.dumps(witness_payload, indent=2) + '\n',
                encoding='utf-8',
            )
            local_root = output / 'local_lightcones'
            local_root.mkdir(parents=True, exist_ok=True)
            copied_features = []
            for feature in local_manifest['active_features']:
                source_qasm = Path(feature['qasm_path'])
                destination = local_root / source_qasm.name
                shutil.copy2(source_qasm, destination)
                copied = dict(feature)
                copied['qasm_path'] = str(destination)
                copied_features.append(copied)
            local_manifest['active_features'] = copied_features
            local_manifest['execution_role'] = (
                'probe-construction audit only; diagnostic and final '
                'certification execute the full 48-qubit science circuit'
            )
            (output / 'local_lightcone_manifest.json').write_text(
                json.dumps(local_manifest, indent=2) + '\n',
                encoding='utf-8',
            )

        prepared_gates = prepared.get('gates', {})
        qpu_gates = {
            'schema': 'codegap.qpu-gates.v3',
            'A_CODE_STATIC': prepared_gates.get('A_CODE_STATIC'),
            'A_CODE_QPU': prepared_gates.get('A_CODE_QPU'),
            'B_NOISE_OFFLINE': prepared_gates.get('B_NOISE'),
            'B_NOISE_QPU': {
                'passed': True,
                'evidence': selected,
            },
            'C_HARDNESS_GAP': prepared_gates.get('C_HARDNESS'),
            'D_QPU_DIAGNOSTIC': {'passed': False, 'status': 'NOT_RUN'},
            'E_QPU_FINAL': {'passed': False, 'status': 'NOT_RUN'},
        }
        (output / 'qpu_gate_report.json').write_text(
            json.dumps(qpu_gates, indent=2) + '\n',
            encoding='utf-8',
        )
    freeze(output)
    return report


def submit_science(
    *,
    selected_root: Path,
    campaign_root: Path,
    credentials: Path | None,
    stage: str,
    shots: int,
    output: Path,
) -> dict:
    if stage not in {'diagnostic', 'final'}:
        raise ValueError('stage must be diagnostic or final.')
    selected_root = selected_root.resolve()
    campaign_root = campaign_root.resolve()
    output = output.resolve()

    selected_freeze = selected_root / 'freeze_manifest.json'
    if not selected_freeze.is_file() or not verify_freeze(selected_freeze)['ok']:
        raise RuntimeError('Selected QPU bundle freeze verification failed.')
    campaign_freeze = campaign_root / 'freeze_manifest.json'
    if not campaign_freeze.is_file() or not verify_freeze(campaign_freeze)['ok']:
        raise RuntimeError('Campaign freeze verification failed.')

    selection = _load(selected_root / 'qpu_selection.json')
    if selection['status'] != 'PASS':
        raise RuntimeError('QPU selection did not pass.')
    campaign = _load(campaign_root / 'campaign_manifest.json')
    config = _load(Path(campaign['source_artifact']) / 'resolved_config.json')
    required_stage_shots = int(config['qpu'][f'{stage}_shots'])
    if shots < required_stage_shots:
        raise RuntimeError(
            f'{stage} shots are below the preregistered budget '
            f'{required_stage_shots}.'
        )

    if stage == 'final':
        diagnostic_path = selected_root / 'diagnostic_certificate.json'
        if not diagnostic_path.is_file():
            raise RuntimeError(
                'Final QPU execution requires a frozen PASS diagnostic certificate.'
            )
        diagnostic = _load(diagnostic_path)
        if not diagnostic.get('pass'):
            raise RuntimeError(
                'Final QPU execution requires a frozen PASS diagnostic certificate.'
            )

    expected_snapshot = _load(campaign_root / 'target_snapshot.json')
    output.mkdir(parents=True, exist_ok=True)
    with OpenQuantumProvider(credentials) as provider:
        backend = provider.backend(
            campaign['backend'],
            campaign['execution_plan'],
            campaign['queue_priority'],
        )
        current_snapshot = snapshot_target(
            backend,
            backend_name=campaign['backend'],
        )
        if (
            structural_fingerprint(current_snapshot)
            != expected_snapshot['structural_fingerprint']
        ):
            raise RuntimeError(
                'Backend target structure changed after preregistration.'
            )
        write_snapshot(
            current_snapshot,
            output / 'submission_target_snapshot.json',
        )
        result = provider.submit_bundle(
            backend_name=campaign['backend'],
            qasm_files=[selected_root / 'science.qasm3'],
            labels=['science'],
            shots=shots,
            output=output,
            stage=stage,
            execution_plan=campaign['execution_plan'],
            queue_priority=campaign['queue_priority'],
        )
    freeze(output)
    return result


def certify_science(
    *,
    selected_root: Path,
    science_run: Path,
    stage: str,
    output: Path,
) -> dict:
    if stage not in {'diagnostic', 'final'}:
        raise ValueError('stage must be diagnostic or final.')
    selected_root = selected_root.resolve()
    science_run = science_run.resolve()
    output = output.resolve()

    selected_freeze = selected_root / 'freeze_manifest.json'
    if not selected_freeze.is_file() or not verify_freeze(selected_freeze)['ok']:
        raise RuntimeError(
            'Selected QPU bundle freeze verification failed before certification.'
        )
    run_freeze = science_run / 'freeze_manifest.json'
    if not run_freeze.is_file() or not verify_freeze(run_freeze)['ok']:
        raise RuntimeError(
            'Science run must be frozen and verified before certification.'
        )

    selection = _load(selected_root / 'qpu_selection.json')
    if selection['status'] != 'PASS':
        raise RuntimeError('QPU selection did not pass.')
    selected = selection['selected']
    raw = _load(science_run / 'raw_counts.json')
    if raw.get('stage') != stage:
        raise RuntimeError(
            'Science run stage does not match certification stage.'
        )
    if raw.get('backend') != _load(
        Path(selection['campaign_root']) / 'campaign_manifest.json'
    )['backend']:
        raise RuntimeError(
            'Science result backend does not match the preregistered campaign.'
        )

    qpu_config = selection['qpu_config']
    stage_config = qpu_config[stage]
    certificate = certify_qpu_counts(
        raw_counts=raw,
        label='science',
        convention=selected['convention'],
        witness_path=selected_root / 'robust_witness.json',
        alpha=float(stage_config['alpha']),
        adversary_generalization_penalty=float(
            stage_config['adversary_generalization_penalty']
        ),
    )
    result_entry = next(
        item for item in raw['results'] if item['label'] == 'science'
    )
    shots_received = int(result_entry['shots_received'])
    required_shots = int(qpu_config[f'{stage}_shots'])
    threshold = float(stage_config['minimum_margin_lcb'])
    certificate.update(
        {
            'stage': stage,
            'shots_received': shots_received,
            'required_shots': required_shots,
            'minimum_margin_lcb': threshold,
            'pass': bool(
                shots_received >= required_shots
                and certificate['margin_lcb'] > threshold
            ),
            'probe_selection_sha256': _digest(
                selected_root / 'qpu_selection.json'
            ),
            'science_run_sha256': _digest(
                science_run / 'raw_counts.json'
            ),
        }
    )

    output.mkdir(parents=True, exist_ok=True)
    certificate_path = output / f'{stage}_certificate.json'
    certificate_path.write_text(
        json.dumps(certificate, indent=2),
        encoding='utf-8',
    )

    gate_path = selected_root / 'qpu_gate_report.json'
    gates = (
        _load(gate_path)
        if gate_path.is_file()
        else {'schema': 'codegap.qpu-gates.v1'}
    )
    gate_name = (
        'D_QPU_DIAGNOSTIC'
        if stage == 'diagnostic'
        else 'E_QPU_FINAL'
    )
    gates[gate_name] = {
        'passed': certificate['pass'],
        'status': 'PASS' if certificate['pass'] else 'FAIL',
        'evidence': certificate,
    }
    gate_path.write_text(
        json.dumps(gates, indent=2),
        encoding='utf-8',
    )
    destination_name = (
        'diagnostic_certificate.json'
        if stage == 'diagnostic'
        else 'final_certificate.json'
    )
    shutil.copy2(certificate_path, selected_root / destination_name)
    freeze(selected_root)
    freeze(output)
    return certificate
