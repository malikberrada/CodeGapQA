from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import traceback
import time
from typing import Any


_QUBIT_REGISTER_RE = re.compile(r'(?m)^\s*qubit(?:\s|\[)')
_PHYSICAL_QUBIT_RE = re.compile(r'\$(\d+)')



_BIT_DECL_RE = re.compile(r'(?m)^\s*bit\s*\[\s*(\d+)\s*\]')
_CREG_DECL_RE = re.compile(r'(?m)^\s*creg\s+\w+\s*\[\s*(\d+)\s*\]')
_BIT_KEY_RE = re.compile(r'^(?:0x[0-9a-fA-F]+|[01]+|[0-9]+)$')


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def _classical_width(source: str) -> int:
    match = _BIT_DECL_RE.search(source) or _CREG_DECL_RE.search(source)
    if match is None:
        raise RuntimeError('Could not determine the classical output width from QASM.')
    return int(match.group(1))


def _normalise_count_key(key: Any, *, width: int) -> str:
    token = str(key).replace(' ', '').replace('_', '')
    if token.lower().startswith('0x'):
        return format(int(token, 16), f'0{width}b')
    if set(token) <= {'0', '1'} and token:
        return token.zfill(width)
    if token.isdigit():
        return format(int(token), f'0{width}b')
    raise ValueError(f'Unsupported count key: {key!r}.')


def _count_maps(payload: Any):
    if isinstance(payload, dict):
        if payload and all(
            _BIT_KEY_RE.fullmatch(str(key).replace(' ', ''))
            and isinstance(value, (int, float))
            and float(value).is_integer()
            and int(value) >= 0
            for key, value in payload.items()
        ):
            yield payload
        for key in ('counts', 'histogram', 'result', 'results', 'data', 'output'):
            if key in payload:
                yield from _count_maps(payload[key])
        for key, value in payload.items():
            if key not in {'counts', 'histogram', 'result', 'results', 'data', 'output'}:
                yield from _count_maps(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _count_maps(value)


def _extract_counts(payload: Any, *, shots: int, width: int) -> dict[str, int]:
    candidates = []
    for mapping in _count_maps(payload):
        try:
            counts: dict[str, int] = {}
            for key, value in mapping.items():
                normalised = _normalise_count_key(key, width=width)
                counts[normalised] = counts.get(normalised, 0) + int(value)
        except (TypeError, ValueError):
            continue
        total = sum(counts.values())
        candidates.append((total == shots, total, counts))
    exact = [item for item in candidates if item[0]]
    if exact:
        exact.sort(key=lambda item: (len(item[2]), item[1]), reverse=True)
        return exact[0][2]
    totals = sorted({item[1] for item in candidates})
    raise RuntimeError(
        f'Could not find a count map with exactly {shots} shots in the Core SDK output. '
        f'Observed candidate totals: {totals}.'
    )


def _job_field(job: Any, name: str, default: Any = None) -> Any:
    value = getattr(job, name, default)
    if callable(value):
        try:
            return value()
        except Exception:
            return default
    return value


def _job_status(job: Any) -> str:
    return str(_job_field(job, 'status', '')).strip()


def _job_id(job: Any) -> str | None:
    value = _job_field(job, 'id') or _job_field(job, 'job_id')
    return None if value is None else str(value)


def _core_imports():
    try:
        from openquantum_sdk.clients import JobSubmissionConfig
        from openquantum_sdk.enums import ExecutionPlanType, QueuePriorityType
    except ImportError as error:
        raise RuntimeError(
            'Direct physical-QASM submission requires the Open Quantum Core SDK.'
        ) from error
    return JobSubmissionConfig, ExecutionPlanType, QueuePriorityType


def _core_async_imports():
    try:
        from openquantum_sdk.clients import JobSubmissionConfig
        from openquantum_sdk.enums import ExecutionPlanType, QueuePriorityType
        from openquantum_sdk.models import JobPreparationCreate, JobCreate
    except ImportError as error:
        raise RuntimeError(
            'Asynchronous physical-QASM submission requires the Open Quantum Core SDK.'
        ) from error
    return (
        JobSubmissionConfig,
        ExecutionPlanType,
        QueuePriorityType,
        JobPreparationCreate,
        JobCreate,
    )


def _terminal_job_status(status: Any) -> bool:
    return str(status).strip().lower() in {'completed', 'failed', 'cancelled', 'canceled'}


def _successful_job_status(status: Any) -> bool:
    return str(status).strip().lower() == 'completed'


def _coerce_sdk_choice(value: Any, enum_type: Any, *, field: str) -> Any:
    """Convert CLI/config strings to the Core SDK enum expected by submit_job()."""
    if isinstance(value, enum_type):
        return value
    token = str(value).strip().lower()
    if token == 'auto':
        return 'auto'
    for member in enum_type:
        if token in {str(member.name).lower(), str(member.value).lower()}:
            return member
    allowed = ['auto'] + sorted(str(member.name).lower() for member in enum_type)
    raise ValueError(
        f'Unsupported {field} {value!r}. Expected one of: {allowed}.'
    )

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _imports():
    try:
        from openquantum_sdk.auth import ClientCredentials
        from openquantum_sdk_qiskit import OpenQuantumService
        from qiskit import QuantumCircuit, qasm3
    except ImportError as error:
        raise RuntimeError(
            'QPU support requires pip install -e "[qpu]". '
            'The Open Quantum plugin currently requires qiskit>=2.0.'
        ) from error
    return ClientCredentials, OpenQuantumService, qasm3, QuantumCircuit


def _qasm_addressing(source: str) -> str:
    has_register = bool(_QUBIT_REGISTER_RE.search(source))
    has_physical = bool(_PHYSICAL_QUBIT_RE.search(source))
    if has_register and has_physical:
        return 'mixed'
    if has_physical:
        return 'physical'
    if has_register:
        return 'register'
    return 'missing'


def _validate_submission_qasm(source: str, *, backend_num_qubits: int) -> dict[str, Any]:
    addressing = _qasm_addressing(source)
    if addressing == 'missing':
        raise RuntimeError(
            'Submission QASM has neither a qubit register declaration nor '
            'physical-qubit identifiers ($N). Refusing provider submission.'
        )
    physical_ids = [int(item) for item in _PHYSICAL_QUBIT_RE.findall(source)]
    if physical_ids and max(physical_ids) >= backend_num_qubits:
        raise RuntimeError(
            f'Submission QASM references physical qubit ${max(physical_ids)}, '
            f'but backend width is {backend_num_qubits}.'
        )
    return {
        'addressing': addressing,
        'physical_qubits': sorted(set(physical_ids)),
        'backend_num_qubits': int(backend_num_qubits),
    }


def _backend_width(backend: Any) -> int:
    for owner in (backend, getattr(backend, 'target', None)):
        if owner is None:
            continue
        for name in ('num_qubits', 'num_qubits'):
            value = getattr(owner, name, None)
            if value is not None:
                resolved = value() if callable(value) else value
                if resolved is not None and int(resolved) > 0:
                    return int(resolved)
    raise RuntimeError('Could not determine backend qubit width.')


def _canonical_register_circuit(circuit: Any, *, num_qubits: int, QuantumCircuit: Any) -> Any:
    """Rebuild a circuit over one canonical full-width q register.

    This is the compatibility fallback for Qiskit versions that do not expose
    QuantumCircuit.ensure_physical(). Qubit indices in imported physical QASM
    are already hardware indices; preserving those indices in a full-width
    q register gives an explicit identity physical mapping.
    """
    if int(circuit.num_qubits) > int(num_qubits):
        raise RuntimeError(
            f'Circuit width {circuit.num_qubits} exceeds backend width {num_qubits}.'
        )
    rebuilt = QuantumCircuit(int(num_qubits), int(circuit.num_clbits), name=circuit.name)
    rebuilt.global_phase = circuit.global_phase
    metadata = getattr(circuit, 'metadata', None)
    rebuilt.metadata = dict(metadata) if isinstance(metadata, dict) else metadata
    for instruction in circuit.data:
        qargs = [rebuilt.qubits[circuit.find_bit(bit).index] for bit in instruction.qubits]
        cargs = [rebuilt.clbits[circuit.find_bit(bit).index] for bit in instruction.clbits]
        rebuilt.append(instruction.operation, qargs, cargs)
    return rebuilt


def _load_submission_circuit(
    *, path: Path, qasm3: Any, QuantumCircuit: Any, backend_num_qubits: int
) -> tuple[Any, str, dict[str, Any]]:
    source = path.read_text(encoding='utf-8')
    source_validation = _validate_submission_qasm(
        source, backend_num_qubits=backend_num_qubits
    )
    try:
        circuit = qasm3.loads(source, num_qubits=int(backend_num_qubits))
        importer_mode = 'qasm3.loads(num_qubits=backend_width)'
    except TypeError:
        circuit = qasm3.loads(source)
        importer_mode = 'qasm3.loads(legacy)'

    ensure_physical = getattr(circuit, 'ensure_physical', None)
    physicalization = None
    if callable(ensure_physical):
        try:
            if int(circuit.num_qubits) < int(backend_num_qubits):
                ensure_physical(num_qubits=int(backend_num_qubits))
            else:
                ensure_physical()
            physicalization = 'QuantumCircuit.ensure_physical'
        except Exception:
            circuit = _canonical_register_circuit(
                circuit,
                num_qubits=int(backend_num_qubits),
                QuantumCircuit=QuantumCircuit,
            )
            physicalization = 'canonical_full_width_register_fallback'
    else:
        circuit = _canonical_register_circuit(
            circuit,
            num_qubits=int(backend_num_qubits),
            QuantumCircuit=QuantumCircuit,
        )
        physicalization = 'canonical_full_width_register_fallback'

    if int(circuit.num_qubits) != int(backend_num_qubits):
        circuit = _canonical_register_circuit(
            circuit,
            num_qubits=int(backend_num_qubits),
            QuantumCircuit=QuantumCircuit,
        )
        physicalization = 'canonical_full_width_register_fallback'

    submission_source = qasm3.dumps(circuit)
    submission_validation = _validate_submission_qasm(
        submission_source, backend_num_qubits=backend_num_qubits
    )
    if (
        source_validation['addressing'] == 'physical'
        and submission_validation['addressing'] == 'register'
        and int(circuit.num_qubits) != int(backend_num_qubits)
    ):
        raise RuntimeError(
            'Physical QASM was converted to a partial virtual register, which '
            'would lose the pinned hardware mapping.'
        )
    diagnostics = {
        'source_addressing': source_validation['addressing'],
        'submission_addressing': submission_validation['addressing'],
        'source_physical_qubits': source_validation['physical_qubits'],
        'submission_physical_qubits': submission_validation['physical_qubits'],
        'backend_num_qubits': int(backend_num_qubits),
        'importer_mode': importer_mode,
        'physicalization': physicalization,
    }
    return circuit, submission_source, diagnostics


def load_credentials(path: Path | None) -> Any:
    ClientCredentials, _, _, _ = _imports()
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    return ClientCredentials(
        client_id=payload['client_id'],
        client_secret=payload['client_secret'],
    )


class OpenQuantumProvider:
    def __init__(self, credentials: Path | None = None):
        _, OpenQuantumService, _, _ = _imports()
        creds = load_credentials(credentials)
        self.service = OpenQuantumService(creds=creds) if creds is not None else OpenQuantumService()

    def close(self) -> None:
        self.service.close()

    def __enter__(self) -> 'OpenQuantumProvider':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def backend(self, name: str, execution_plan: str = 'public', queue_priority: str = 'standard') -> Any:
        return self.service.return_backend(
            name,
            export_format='qasm3',
            config={
                'job_subcategory_id': 'phys:oth',
                'execution_plan': execution_plan,
                'queue_priority': queue_priority,
            },
        )

    def backend_metadata(self, name: str) -> dict[str, Any]:
        matches = self.service.backends(name=name)
        for item in matches:
            if item.get('short_code') == name or item.get('name') == name:
                return dict(item)
        return dict(matches[0]) if matches else {}

    def _scheduler(self) -> Any:
        scheduler = getattr(self.service, 'scheduler', None)
        if scheduler is None:
            scheduler = getattr(self.service, '_scheduler', None)
        if scheduler is None:
            raise RuntimeError('OpenQuantumService does not expose a SchedulerClient.')
        return scheduler

    def list_jobs(self, *, status: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        scheduler = self._scheduler()
        response = scheduler.list_jobs(status=status) if status else scheduler.list_jobs()
        jobs = list(getattr(response, 'jobs', []) or [])[: max(1, int(limit))]
        output = []
        for job in jobs:
            output.append(
                {
                    'id': _job_id(job),
                    'name': str(_job_field(job, 'name', '')),
                    'status': _job_status(job),
                    'backend_class_id': _job_field(job, 'backend_class_id'),
                    'submitted_at': str(_job_field(job, 'submitted_at', '')),
                    'completed_at': str(_job_field(job, 'completed_at', '')),
                    'credits_used': _job_field(job, 'credits_used'),
                    'shots': _job_field(job, 'shots'),
                }
            )
        return output

    def _existing_job(self, scheduler: Any, name: str) -> Any | None:
        try:
            response = scheduler.list_jobs()
        except Exception:
            return None
        for job in list(getattr(response, 'jobs', []) or []):
            if str(_job_field(job, 'name', '')) == name:
                return job
        return None

    def _wait_existing_job(self, scheduler: Any, job: Any, *, timeout: int = 86400) -> Any:
        deadline = time.monotonic() + timeout
        current = job
        while _job_status(current).lower() not in {'completed', 'failed', 'cancelled'}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timed out waiting for Open Quantum job {_job_id(current)}.')
            time.sleep(10)
            current = scheduler.get_job(_job_id(current))
        return current

    def submit_bundle(
        self,
        *,
        backend_name: str,
        qasm_files: list[Path],
        labels: list[str],
        shots: int,
        output: Path,
        stage: str,
        execution_plan: str = 'public',
        queue_priority: str = 'standard',
        prepare_only: bool = False,
    ) -> dict[str, Any]:
        if len(qasm_files) != len(labels):
            raise ValueError('qasm_files and labels must have equal length.')
        if shots <= 0:
            raise ValueError('shots must be positive.')

        output.mkdir(parents=True, exist_ok=True)
        submission_path = output / 'submission.json'
        results_path = output / 'raw_counts.json'
        raw_jobs_root = output / 'raw_jobs'
        submission_qasm_root = output / 'submission_qasm'
        raw_jobs_root.mkdir(parents=True, exist_ok=True)
        submission_qasm_root.mkdir(parents=True, exist_ok=True)

        backend = self.backend(backend_name, execution_plan, queue_priority)
        backend_num_qubits = _backend_width(backend)
        expected = []
        for index, (label, path) in enumerate(zip(labels, qasm_files)):
            source = path.read_text(encoding='utf-8')
            validation = _validate_submission_qasm(source, backend_num_qubits=backend_num_qubits)
            if validation['addressing'] != 'physical':
                raise RuntimeError(
                    f'Probe {label!r} is not explicit physical QASM. Expected $N identifiers; '
                    f'observed addressing={validation["addressing"]!r}.'
                )
            safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', label).strip('_') or f'circuit_{index}'
            copied = submission_qasm_root / f'{index:03d}_{safe_label}.qasm3'
            copied.write_text(source, encoding='utf-8')
            digest = sha256_file(copied)
            job_name = f'CodeGap-{stage}-{index:03d}-{digest[:10]}'
            expected.append(
                {
                    'index': index,
                    'label': label,
                    'job_name': job_name,
                    'qasm_path': str(path.resolve()),
                    'qasm_sha256': sha256_file(path),
                    'submission_qasm_path': str(copied.resolve()),
                    'submission_qasm_sha256': digest,
                    'addressing': 'physical',
                    'physical_qubits': validation['physical_qubits'],
                    'classical_bits': _classical_width(source),
                    'status': 'PENDING',
                    'job_id': None,
                    'attempts': 0,
                }
            )

        if submission_path.exists():
            submission = json.loads(submission_path.read_text(encoding='utf-8'))
            if submission.get('schema') != 'codegap.qpu-submission.v3-core-raw-resumable':
                raise RuntimeError(
                    'The output directory contains a non-resumable pre-v0.5.7 submission. '
                    'Use a new directory, or import the completed portal job first.'
                )
            invariant = (
                submission.get('backend') == backend_name
                and submission.get('stage') == stage
                and int(submission.get('shots_requested_per_circuit', -1)) == int(shots)
                and [item.get('label') for item in submission.get('circuits', [])] == labels
                and [item.get('submission_qasm_sha256') for item in submission.get('circuits', [])]
                == [item['submission_qasm_sha256'] for item in expected]
            )
            if not invariant:
                raise RuntimeError('Existing resumable submission does not match this campaign.')
        else:
            submission = {
                'schema': 'codegap.qpu-submission.v3-core-raw-resumable',
                'created_at': utc_now(),
                'provider': 'openquantum-core-sdk',
                'transport': 'SchedulerClient.submit_job(file_content=physical_qasm_bytes)',
                'backend': backend_name,
                'backend_num_qubits': backend_num_qubits,
                'stage': stage,
                'shots_requested_per_circuit': shots,
                'execution_plan': execution_plan,
                'queue_priority': queue_priority,
                'status': 'PREPARED',
                'circuits': expected,
            }
            _write_json_atomic(submission_path, submission)

        if prepare_only:
            return {
                'status': 'PREPARED',
                'output': str(output.resolve()),
                'circuits': len(submission['circuits']),
                'completed': sum(
                    1 for item in submission['circuits'] if item.get('status') == 'COMPLETED'
                ),
            }

        scheduler = self._scheduler()
        JobSubmissionConfig, ExecutionPlanType, QueuePriorityType = _core_imports()
        sdk_execution_plan = _coerce_sdk_choice(
            execution_plan, ExecutionPlanType, field='execution_plan'
        )
        sdk_queue_priority = _coerce_sdk_choice(
            queue_priority, QueuePriorityType, field='queue_priority'
        )
        submission['sdk_execution_plan'] = (
            sdk_execution_plan if sdk_execution_plan == 'auto' else sdk_execution_plan.name.lower()
        )
        submission['sdk_queue_priority'] = (
            sdk_queue_priority if sdk_queue_priority == 'auto' else sdk_queue_priority.name.lower()
        )
        submission['status'] = 'RUNNING'
        _write_json_atomic(submission_path, submission)

        for item in submission['circuits']:
            if item.get('status') == 'COMPLETED':
                continue
            raw_path = raw_jobs_root / f'{int(item["index"]):03d}.json'
            try:
                raw_output = None
                job = None
                if raw_path.is_file():
                    raw_output = json.loads(raw_path.read_text(encoding='utf-8'))
                if raw_output is None and item.get('job_id'):
                    job = scheduler.get_job(item['job_id'])
                    job = self._wait_existing_job(scheduler, job)
                    if _job_status(job).lower() != 'completed':
                        raise RuntimeError(
                            f'Open Quantum job {item["job_id"]} ended as {_job_status(job)}.'
                        )
                    raw_output = scheduler.download_job_output(job)
                if raw_output is None:
                    existing = self._existing_job(scheduler, item['job_name'])
                    if existing is not None:
                        existing = self._wait_existing_job(scheduler, existing)
                        if _job_status(existing).lower() == 'completed':
                            job = existing
                            item['job_id'] = _job_id(job)
                            item['status'] = 'DOWNLOADING_EXISTING'
                            _write_json_atomic(submission_path, submission)
                            raw_output = scheduler.download_job_output(job)
                if raw_output is None:
                    item['attempts'] = int(item.get('attempts', 0)) + 1
                    item['status'] = 'SUBMITTING'
                    _write_json_atomic(submission_path, submission)
                    config = JobSubmissionConfig(
                        backend_class_id=backend_name,
                        name=item['job_name'],
                        job_subcategory_id='phys:oth',
                        shots=int(shots),
                        execution_plan=sdk_execution_plan,
                        queue_priority=sdk_queue_priority,
                        auto_approve_quote=True,
                        verbose=True,
                    )
                    source = Path(item['submission_qasm_path']).read_bytes()
                    job = scheduler.submit_job(config, file_content=source)
                    item['job_id'] = _job_id(job)
                    item['status'] = 'DOWNLOADING'
                    item['completed_job_status'] = _job_status(job)
                    _write_json_atomic(submission_path, submission)
                    raw_output = scheduler.download_job_output(job)

                _write_json_atomic(raw_path, raw_output)
                counts = _extract_counts(
                    raw_output, shots=int(shots), width=int(item['classical_bits'])
                )
                item.update(
                    {
                        'status': 'COMPLETED',
                        'completed_at': utc_now(),
                        'raw_output_path': str(raw_path.resolve()),
                        'shots_received': sum(counts.values()),
                        'distinct_outcomes': len(counts),
                        'counts': counts,
                    }
                )
                _write_json_atomic(submission_path, submission)
            except Exception as error:
                item['status'] = 'ERROR'
                item['error_type'] = type(error).__name__
                item['error'] = str(error)
                submission['status'] = 'PARTIAL_ERROR'
                _write_json_atomic(submission_path, submission)
                _write_json_atomic(
                    output / 'error.json',
                    {
                        'schema': 'codegap.qpu-error.v2-resumable',
                        'failed_at': utc_now(),
                        'label': item['label'],
                        'job_id': item.get('job_id'),
                        'error_type': type(error).__name__,
                        'error': str(error),
                        'traceback': traceback.format_exc(),
                        'resume_command_safe': True,
                    },
                )
                raise

        bundles = [
            {
                'label': item['label'],
                'result_register': 'core_sdk_counts',
                'classical_bits': int(item['classical_bits']),
                'shots_received': int(item['shots_received']),
                'distinct_outcomes': int(item['distinct_outcomes']),
                'counts': dict(item['counts']),
                'job_id': item.get('job_id'),
            }
            for item in submission['circuits']
        ]
        payload = {
            'schema': 'codegap.qpu-counts.v2-resumable',
            'completed_at': utc_now(),
            'provider': 'openquantum-core-sdk',
            'backend': backend_name,
            'stage': stage,
            'shots_per_circuit': shots,
            'results': bundles,
        }
        _write_json_atomic(results_path, payload)
        submission.update(
            {
                'status': 'COMPLETED',
                'completed_at': utc_now(),
                'results_path': str(results_path.resolve()),
                'completed_circuits': len(bundles),
            }
        )
        _write_json_atomic(submission_path, submission)
        error_path = output / 'error.json'
        if error_path.exists():
            error_path.unlink()
        return payload

    def _resolve_organization_id(self) -> str:
        management = getattr(self.service, 'management', None)
        if management is None:
            management = getattr(self.service, '_management', None)
        if management is None:
            raise RuntimeError('OpenQuantumService does not expose a ManagementClient.')
        response = management.list_user_organizations()
        organizations = list(
            getattr(response, 'organizations', None)
            or getattr(response, 'items', None)
            or []
        )
        if not organizations and isinstance(response, dict):
            organizations = list(
                response.get('organizations') or response.get('items') or []
            )
        if not organizations:
            raise RuntimeError('No Open Quantum organization is available for this SDK key.')

        def field(item: Any, name: str, default: Any = None) -> Any:
            if isinstance(item, dict):
                return item.get(name, default)
            return getattr(item, name, default)

        preferred = [
            item
            for item in organizations
            if bool(field(item, 'is_default', False))
            or bool(field(item, 'is_active', False))
            or bool(field(item, 'default', False))
        ]
        chosen = preferred[0] if len(preferred) == 1 else None
        if chosen is None and len(organizations) == 1:
            chosen = organizations[0]
        if chosen is None:
            ids = [str(field(item, 'id', '')) for item in organizations]
            raise RuntimeError(
                'Multiple Open Quantum organizations are available and none is marked '
                f'as default. Organization IDs: {ids}.'
            )
        organization_id = field(chosen, 'id') or field(chosen, 'organization_id')
        if not organization_id:
            raise RuntimeError('Could not resolve the Open Quantum organization ID.')
        return str(organization_id)

    def _async_submission_state(
        self,
        *,
        backend_name: str,
        qasm_files: list[Path],
        labels: list[str],
        shots: int,
        output: Path,
        stage: str,
        execution_plan: str,
        queue_priority: str,
    ) -> tuple[dict[str, Any], Path]:
        submission_path = output / 'submission.json'
        if not submission_path.is_file():
            self.submit_bundle(
                backend_name=backend_name,
                qasm_files=qasm_files,
                labels=labels,
                shots=shots,
                output=output,
                stage=stage,
                execution_plan=execution_plan,
                queue_priority=queue_priority,
                prepare_only=True,
            )
        submission = json.loads(submission_path.read_text(encoding='utf-8'))
        schema = submission.get('schema')
        if schema == 'codegap.qpu-submission.v3-core-raw-resumable':
            submission['schema'] = 'codegap.qpu-submission.v4-core-async'
            submission['transport'] = (
                'SchedulerClient.upload_job_input/prepare_job/create_job '
                '(asynchronous physical QASM)'
            )
            submission['async_submission'] = True
            submission['status'] = (
                'COMPLETED'
                if all(row.get('status') == 'COMPLETED' for row in submission['circuits'])
                else 'READY_TO_SUBMIT'
            )
            _write_json_atomic(submission_path, submission)
        elif schema != 'codegap.qpu-submission.v4-core-async':
            raise RuntimeError(
                f'Unsupported probe submission schema {schema!r}; use a fresh output directory.'
            )
        invariant = (
            submission.get('backend') == backend_name
            and submission.get('stage') == stage
            and int(submission.get('shots_requested_per_circuit', -1)) == int(shots)
            and [row.get('label') for row in submission.get('circuits', [])] == labels
        )
        if not invariant:
            raise RuntimeError('Existing asynchronous submission does not match this campaign.')
        return submission, submission_path

    def attach_async_job(
        self,
        *,
        backend_name: str,
        qasm_files: list[Path],
        labels: list[str],
        shots: int,
        output: Path,
        stage: str,
        execution_plan: str,
        queue_priority: str,
        job_id: str,
        label: str | None = None,
    ) -> dict[str, Any]:
        submission, submission_path = self._async_submission_state(
            backend_name=backend_name,
            qasm_files=qasm_files,
            labels=labels,
            shots=shots,
            output=output,
            stage=stage,
            execution_plan=execution_plan,
            queue_priority=queue_priority,
        )
        if label is None:
            candidates = [
                row
                for row in submission['circuits']
                if row.get('status') != 'COMPLETED' and not row.get('job_id')
            ]
            if not candidates:
                raise RuntimeError('No incomplete probe is available for automatic job attachment.')
            item = candidates[0]
        else:
            matches = [row for row in submission['circuits'] if row['label'] == label]
            if len(matches) != 1:
                raise RuntimeError(f'Expected exactly one probe labelled {label!r}.')
            item = matches[0]
        scheduler = self._scheduler()
        job = scheduler.get_job(str(job_id))
        status = _job_status(job)
        if status.strip().lower() in {'failed', 'cancelled', 'canceled'}:
            raise RuntimeError(f'Cannot attach Open Quantum job {job_id}: status={status}.')
        portal_name = str(_job_field(job, 'name', '') or '')
        item.update(
            {
                'job_id': str(job_id),
                'status': 'SUBMITTED' if not _successful_job_status(status) else 'REMOTE_COMPLETED',
                'portal_status': status,
                'portal_name': portal_name,
                'attached_at': utc_now(),
                'attached_manually': True,
                'error': None,
                'error_type': None,
            }
        )
        submission['status'] = 'PARTIALLY_SUBMITTED'
        _write_json_atomic(submission_path, submission)
        return {
            'schema': 'codegap.qpu-attach-job.v1',
            'status': 'ATTACHED',
            'label': item['label'],
            'job_id': str(job_id),
            'portal_status': status,
            'remaining_without_job_id': sum(
                1
                for row in submission['circuits']
                if row.get('status') != 'COMPLETED' and not row.get('job_id')
            ),
        }

    def _wait_for_preparation_result(
        self,
        scheduler: Any,
        preparation_id: str,
        *,
        poll_seconds: int = 3,
        timeout: int = 1800,
    ) -> Any:
        deadline = time.monotonic() + int(timeout)
        while True:
            result = scheduler.get_preparation_result(preparation_id)
            status = str(_job_field(result, 'status', '')).strip()
            if status.lower() == 'completed':
                return result
            if status.lower() == 'failed':
                message = _job_field(result, 'message', '')
                raise RuntimeError(f'Job preparation failed: {message}')
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f'Timed out waiting for preparation {preparation_id}.'
                )
            time.sleep(max(1, int(poll_seconds)))

    def _active_portal_jobs(self, scheduler: Any) -> int:
        response = scheduler.list_jobs()
        jobs = list(getattr(response, 'jobs', []) or [])
        return sum(1 for job in jobs if not _terminal_job_status(_job_status(job)))

    def _create_async_job(
        self,
        *,
        scheduler: Any,
        item: dict[str, Any],
        submission: dict[str, Any],
        submission_path: Path,
        backend_name: str,
        shots: int,
        execution_plan: str,
        queue_priority: str,
        preparation_poll_seconds: int,
    ) -> Any:
        (
            JobSubmissionConfig,
            ExecutionPlanType,
            QueuePriorityType,
            JobPreparationCreate,
            JobCreate,
        ) = _core_async_imports()
        sdk_execution_plan = _coerce_sdk_choice(
            execution_plan, ExecutionPlanType, field='execution_plan'
        )
        sdk_queue_priority = _coerce_sdk_choice(
            queue_priority, QueuePriorityType, field='queue_priority'
        )
        config = JobSubmissionConfig(
            backend_class_id=backend_name,
            name=item['job_name'],
            job_subcategory_id='phys:oth',
            shots=int(shots),
            execution_plan=sdk_execution_plan,
            queue_priority=sdk_queue_priority,
            auto_approve_quote=True,
            verbose=False,
        )
        existing = self._existing_job(scheduler, item['job_name'])
        if existing is not None:
            item.update(
                {
                    'job_id': _job_id(existing),
                    'portal_status': _job_status(existing),
                    'status': 'SUBMITTED',
                    'recovered_by_job_name': True,
                }
            )
            _write_json_atomic(submission_path, submission)
            return existing

        organization_id = submission.get('organization_id')
        if not organization_id:
            organization_id = self._resolve_organization_id()
            submission['organization_id'] = organization_id
            _write_json_atomic(submission_path, submission)

        preparation_id = item.get('preparation_id')
        if not preparation_id:
            item['status'] = 'UPLOADING'
            item['attempts'] = int(item.get('attempts', 0)) + 1
            _write_json_atomic(submission_path, submission)
            source = Path(item['submission_qasm_path']).read_bytes()
            upload_id = scheduler.upload_job_input(file_content=source)
            item['upload_endpoint_id'] = str(upload_id)
            item['status'] = 'PREPARING'
            _write_json_atomic(submission_path, submission)
            preparation = JobPreparationCreate(
                organization_id=organization_id,
                backend_class_id=backend_name,
                name=item['job_name'],
                upload_endpoint_id=str(upload_id),
                job_subcategory_id='phys:oth',
                shots=int(shots),
                input_format='qasm',
            )
            preparation_response = scheduler.prepare_job(preparation)
            preparation_id = str(_job_field(preparation_response, 'id'))
            if not preparation_id or preparation_id == 'None':
                raise RuntimeError('Open Quantum preparation returned no ID.')
            item['preparation_id'] = preparation_id
            _write_json_atomic(submission_path, submission)

        preparation_result = self._wait_for_preparation_result(
            scheduler,
            preparation_id,
            poll_seconds=preparation_poll_seconds,
        )
        chooser = getattr(scheduler, '_choose_plan_and_priority', None)
        if not callable(chooser):
            raise RuntimeError(
                'Installed Open Quantum SDK lacks _choose_plan_and_priority; '
                'upgrade the SDK before asynchronous submission.'
            )
        execution_plan_id, queue_priority_id = chooser(preparation_result, config)
        item.update(
            {
                'execution_plan_id': str(execution_plan_id),
                'queue_priority_id': str(queue_priority_id),
                'status': 'CREATING_JOB',
            }
        )
        _write_json_atomic(submission_path, submission)
        job_create = JobCreate(
            organization_id=organization_id,
            job_preparation_id=preparation_id,
            execution_plan_id=execution_plan_id,
            queue_priority_id=queue_priority_id,
        )
        job = scheduler.create_job(job_create)
        job_id = _job_id(job)
        if not job_id:
            raise RuntimeError('Open Quantum create_job() returned no job ID.')
        item.update(
            {
                'job_id': job_id,
                'status': 'SUBMITTED',
                'portal_status': _job_status(job),
                'submitted_at': utc_now(),
                'error': None,
                'error_type': None,
            }
        )
        _write_json_atomic(submission_path, submission)
        return job

    def submit_bundle_async(
        self,
        *,
        backend_name: str,
        qasm_files: list[Path],
        labels: list[str],
        shots: int,
        output: Path,
        stage: str,
        execution_plan: str = 'public',
        queue_priority: str = 'standard',
        max_active: int = 10,
        slot_poll_seconds: int = 15,
        slot_timeout_seconds: int = 86400,
        preparation_poll_seconds: int = 3,
    ) -> dict[str, Any]:
        if not 1 <= int(max_active) <= 10:
            raise ValueError('max_active must be between 1 and 10.')
        submission, submission_path = self._async_submission_state(
            backend_name=backend_name,
            qasm_files=qasm_files,
            labels=labels,
            shots=shots,
            output=output,
            stage=stage,
            execution_plan=execution_plan,
            queue_priority=queue_priority,
        )
        scheduler = self._scheduler()
        submission['status'] = 'SUBMITTING_ALL_JOBS'
        submission['max_active_jobs'] = int(max_active)
        _write_json_atomic(submission_path, submission)
        deadline = time.monotonic() + int(slot_timeout_seconds)

        for item in submission['circuits']:
            if item.get('status') == 'COMPLETED' or item.get('job_id'):
                continue
            existing = self._existing_job(scheduler, item['job_name'])
            if existing is not None:
                item.update(
                    {
                        'job_id': _job_id(existing),
                        'portal_status': _job_status(existing),
                        'status': 'SUBMITTED',
                        'recovered_by_job_name': True,
                    }
                )
                _write_json_atomic(submission_path, submission)
                continue
            while self._active_portal_jobs(scheduler) >= int(max_active):
                if time.monotonic() >= deadline:
                    submission['status'] = 'WAITING_FOR_ACTIVE_JOB_SLOT'
                    _write_json_atomic(submission_path, submission)
                    raise TimeoutError(
                        'Timed out waiting for an Open Quantum active-job slot. '
                        'Rerun the same command to continue.'
                    )
                submission['status'] = 'WAITING_FOR_ACTIVE_JOB_SLOT'
                submission['last_slot_check_at'] = utc_now()
                _write_json_atomic(submission_path, submission)
                time.sleep(max(1, int(slot_poll_seconds)))
            try:
                self._create_async_job(
                    scheduler=scheduler,
                    item=item,
                    submission=submission,
                    submission_path=submission_path,
                    backend_name=backend_name,
                    shots=shots,
                    execution_plan=execution_plan,
                    queue_priority=queue_priority,
                    preparation_poll_seconds=preparation_poll_seconds,
                )
            except Exception as error:
                item['status'] = 'ERROR'
                item['error_type'] = type(error).__name__
                item['error'] = str(error)
                submission['status'] = 'PARTIAL_SUBMISSION_ERROR'
                _write_json_atomic(submission_path, submission)
                raise

        missing = [row['label'] for row in submission['circuits'] if not row.get('job_id') and row.get('status') != 'COMPLETED']
        if missing:
            submission['status'] = 'PARTIALLY_SUBMITTED'
        else:
            submission['status'] = 'ALL_JOBS_SUBMITTED'
            submission['all_jobs_submitted_at'] = utc_now()
        _write_json_atomic(submission_path, submission)
        return {
            'schema': 'codegap.qpu-async-submission.v1',
            'status': submission['status'],
            'output': str(output.resolve()),
            'circuits': len(submission['circuits']),
            'completed_locally': sum(1 for row in submission['circuits'] if row.get('status') == 'COMPLETED'),
            'jobs_recorded': sum(1 for row in submission['circuits'] if row.get('job_id')),
            'remaining_without_job_id': len(missing),
            'jobs': [
                {
                    'index': row['index'],
                    'label': row['label'],
                    'status': row.get('status'),
                    'job_id': row.get('job_id'),
                    'portal_status': row.get('portal_status'),
                }
                for row in submission['circuits']
            ],
        }

    def collect_async_bundle(
        self,
        *,
        output: Path,
        wait: bool = False,
        poll_seconds: int = 20,
        timeout_seconds: int = 86400,
        job_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        submission_path = output / 'submission.json'
        if not submission_path.is_file():
            raise RuntimeError('Asynchronous submission.json is missing.')
        submission = json.loads(submission_path.read_text(encoding='utf-8'))
        if submission.get('schema') != 'codegap.qpu-submission.v4-core-async':
            raise RuntimeError('Probe run is not a v0.5.9+ asynchronous submission.')

        requested_job_ids = {
            str(job_id).strip() for job_id in (job_ids or []) if str(job_id).strip()
        }
        if requested_job_ids:
            known_job_ids = {
                str(item.get('job_id'))
                for item in submission['circuits']
                if item.get('job_id')
            }
            unknown = sorted(requested_job_ids - known_job_ids)
            if unknown:
                raise RuntimeError(
                    'Requested job IDs are not attached to this probe run: '
                    + ', '.join(unknown)
                )

        scheduler = self._scheduler()
        raw_jobs_root = output / 'raw_jobs'
        raw_jobs_root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + int(timeout_seconds)

        while True:
            running = []
            selected_pending = 0
            for item in submission['circuits']:
                if item.get('status') == 'COMPLETED':
                    continue
                job_id = item.get('job_id')
                if requested_job_ids and str(job_id or '') not in requested_job_ids:
                    continue
                selected_pending += 1
                if not job_id:
                    raise RuntimeError(
                        f'Probe {item["label"]!r} has no job ID. Run qpu-submit-probes first '
                        'or collect a specific attached job with --job-id.'
                    )
                job = scheduler.get_job(job_id)
                status = _job_status(job)
                item['portal_status'] = status
                item['last_status_check_at'] = utc_now()
                if _successful_job_status(status):
                    raw_output = scheduler.download_job_output(job)
                    raw_path = raw_jobs_root / f'{int(item["index"]):03d}.json'
                    _write_json_atomic(raw_path, raw_output)
                    counts = _extract_counts(
                        raw_output,
                        shots=int(submission['shots_requested_per_circuit']),
                        width=int(item['classical_bits']),
                    )
                    item.update(
                        {
                            'status': 'COMPLETED',
                            'completed_at': utc_now(),
                            'raw_output_path': str(raw_path.resolve()),
                            'shots_received': sum(counts.values()),
                            'distinct_outcomes': len(counts),
                            'counts': counts,
                        }
                    )
                elif status.strip().lower() in {'failed', 'cancelled', 'canceled'}:
                    item['status'] = 'REMOTE_ERROR'
                    submission['status'] = 'REMOTE_JOB_ERROR'
                    _write_json_atomic(submission_path, submission)
                    raise RuntimeError(
                        f'Open Quantum job {job_id} for {item["label"]} ended as {status}.'
                    )
                else:
                    item['status'] = 'SUBMITTED'
                    running.append(item['label'])
                _write_json_atomic(submission_path, submission)

            if not running:
                break
            submission['status'] = (
                'TARGET_JOBS_STILL_RUNNING' if requested_job_ids else 'JOBS_STILL_RUNNING'
            )
            submission['running_labels'] = running
            _write_json_atomic(submission_path, submission)
            if not wait:
                return {
                    'schema': 'codegap.qpu-async-collection.v2',
                    'status': submission['status'],
                    'running': len(running),
                    'completed': sum(
                        1 for row in submission['circuits'] if row.get('status') == 'COMPLETED'
                    ),
                    'running_labels': running,
                }
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    'Timed out waiting for probe jobs. Rerun qpu-collect-probes later.'
                )
            time.sleep(max(1, int(poll_seconds)))

        completed_count = sum(
            1 for row in submission['circuits'] if row.get('status') == 'COMPLETED'
        )
        total_count = len(submission['circuits'])
        if completed_count != total_count:
            submission.update(
                {
                    'status': 'PARTIAL_RESULTS_COLLECTED',
                    'partially_collected_at': utc_now(),
                    'completed_circuits': completed_count,
                    'remaining_circuits': total_count - completed_count,
                    'running_labels': [],
                }
            )
            _write_json_atomic(submission_path, submission)
            return {
                'schema': 'codegap.qpu-async-collection.v2',
                'status': 'PARTIAL_RESULTS_COLLECTED',
                'completed': completed_count,
                'remaining': total_count - completed_count,
                'selected_job_ids': sorted(requested_job_ids),
            }

        bundles = [
            {
                'label': item['label'],
                'result_register': 'core_sdk_counts',
                'classical_bits': int(item['classical_bits']),
                'shots_received': int(item['shots_received']),
                'distinct_outcomes': int(item['distinct_outcomes']),
                'counts': dict(item['counts']),
                'job_id': item.get('job_id'),
            }
            for item in submission['circuits']
        ]
        payload = {
            'schema': 'codegap.qpu-counts.v3-async',
            'completed_at': utc_now(),
            'provider': 'openquantum-core-sdk',
            'backend': submission['backend'],
            'stage': submission['stage'],
            'shots_per_circuit': int(submission['shots_requested_per_circuit']),
            'results': bundles,
        }
        results_path = output / 'raw_counts.json'
        _write_json_atomic(results_path, payload)
        submission.update(
            {
                'status': 'COMPLETED',
                'completed_at': utc_now(),
                'results_path': str(results_path.resolve()),
                'completed_circuits': len(bundles),
                'remaining_circuits': 0,
                'running_labels': [],
            }
        )
        _write_json_atomic(submission_path, submission)
        return {
            'schema': 'codegap.qpu-async-collection.v2',
            'status': 'RESULTS_COLLECTED',
            'completed': len(bundles),
            'results_path': str(results_path.resolve()),
        }

    def import_completed_job(
        self,
        *,
        output: Path,
        job_id: str,
        label: str,
    ) -> dict[str, Any]:
        submission_path = output / 'submission.json'
        if not submission_path.is_file():
            raise RuntimeError('Initialize a v0.5.7 probe run before importing a portal job.')
        submission = json.loads(submission_path.read_text(encoding='utf-8'))
        if submission.get('schema') != 'codegap.qpu-submission.v3-core-raw-resumable':
            raise RuntimeError('Probe run is not a v0.5.7 resumable submission.')
        matches = [item for item in submission['circuits'] if item['label'] == label]
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one circuit labelled {label!r}.')
        item = matches[0]
        scheduler = self._scheduler()
        job = scheduler.get_job(job_id)
        job = self._wait_existing_job(scheduler, job)
        if _job_status(job).lower() != 'completed':
            raise RuntimeError(f'Job {job_id} is not completed: {_job_status(job)}.')
        raw_output = scheduler.download_job_output(job)
        raw_path = output / 'raw_jobs' / f'{int(item["index"]):03d}.json'
        _write_json_atomic(raw_path, raw_output)
        counts = _extract_counts(
            raw_output,
            shots=int(submission['shots_requested_per_circuit']),
            width=int(item['classical_bits']),
        )
        item.update(
            {
                'status': 'COMPLETED',
                'job_id': str(job_id),
                'imported_from_portal': True,
                'completed_at': utc_now(),
                'raw_output_path': str(raw_path.resolve()),
                'shots_received': sum(counts.values()),
                'distinct_outcomes': len(counts),
                'counts': counts,
            }
        )
        _write_json_atomic(submission_path, submission)
        return {
            'status': 'IMPORTED',
            'label': label,
            'job_id': str(job_id),
            'shots_received': sum(counts.values()),
            'remaining': sum(1 for row in submission['circuits'] if row['status'] != 'COMPLETED'),
        }

