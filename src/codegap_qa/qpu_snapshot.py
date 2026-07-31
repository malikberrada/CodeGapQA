from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .qpu_types import TargetInstruction, TargetSnapshot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _property_value(properties: Any, name: str) -> float | None:
    if properties is None:
        return None
    value = getattr(properties, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def snapshot_target(
    backend: Any,
    *,
    backend_name: str | None = None,
    accepting_jobs: bool | None = None,
    queue_depth: int | None = None,
) -> TargetSnapshot:
    target = backend.target
    instructions: list[TargetInstruction] = []
    coupling: set[tuple[int, int]] = set()
    operation_names = tuple(sorted(str(name) for name in target.operation_names))
    for name in operation_names:
        try:
            operation_map = target[name]
        except Exception:
            continue
        if not hasattr(operation_map, 'items'):
            continue
        for raw_qargs, properties in operation_map.items():
            if raw_qargs is None:
                continue
            qargs = tuple(int(value) for value in raw_qargs)
            instructions.append(
                TargetInstruction(
                    name=name,
                    qargs=qargs,
                    error=_property_value(properties, 'error'),
                    duration=_property_value(properties, 'duration'),
                )
            )
            if len(qargs) == 2:
                coupling.add(tuple(sorted(qargs)))
    resolved_name = backend_name or str(getattr(backend, 'name', 'unknown'))
    return TargetSnapshot(
        backend=resolved_name,
        captured_at=utc_now(),
        num_qubits=int(target.num_qubits),
        operation_names=operation_names,
        coupling_edges=tuple(sorted(coupling)),
        instructions=tuple(instructions),
        accepting_jobs=accepting_jobs,
        queue_depth=queue_depth,
    )


def structural_fingerprint(snapshot: TargetSnapshot | dict) -> str:
    payload = snapshot.to_dict() if isinstance(snapshot, TargetSnapshot) else dict(snapshot)
    stable = {
        'backend': payload['backend'],
        'num_qubits': payload['num_qubits'],
        'operation_names': payload['operation_names'],
        'coupling_edges': payload['coupling_edges'],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return sha256(encoded).hexdigest()


def calibration_fingerprint(snapshot: TargetSnapshot | dict) -> str:
    payload = snapshot.to_dict() if isinstance(snapshot, TargetSnapshot) else dict(snapshot)
    stable = {
        'backend': payload['backend'],
        'instructions': payload['instructions'],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return sha256(encoded).hexdigest()


def snapshot_digest(snapshot: TargetSnapshot | dict) -> str:
    payload = snapshot.to_dict() if isinstance(snapshot, TargetSnapshot) else snapshot
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return sha256(encoded).hexdigest()


def write_snapshot(snapshot: TargetSnapshot, path: Path) -> dict:
    payload = snapshot.to_dict()
    payload['sha256'] = snapshot_digest(payload)
    payload['structural_fingerprint'] = structural_fingerprint(payload)
    payload['calibration_fingerprint'] = calibration_fingerprint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload
