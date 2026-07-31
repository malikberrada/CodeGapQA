from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from enum import Enum

import pytest

from codegap_qa import qpu_provider


PHYSICAL_QASM = """OPENQASM 3.0;
bit[2] c;
x $0;
c[0] = measure $0;
c[1] = measure $1;
"""


class FakeJob:
    def __init__(self, job_id: str, name: str, status: str = "Completed"):
        self.id = job_id
        self.name = name
        self.status = status
        self.submitted_at = "2026-07-23T18:00:00Z"
        self.completed_at = "2026-07-23T18:01:00Z"
        self.credits_used = 2
        self.shots = 4


class FakeJobList:
    def __init__(self, jobs):
        self.jobs = list(jobs)


class FakeScheduler:
    def __init__(self, *, fail_on_call: int | None = None):
        self.jobs: list[FakeJob] = []
        self.outputs: dict[str, dict] = {}
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.submitted_bytes: list[bytes] = []

    def list_jobs(self, status=None):
        jobs = self.jobs
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        return FakeJobList(jobs)

    def submit_job(self, config, file_content):
        self.calls += 1
        assert config.execution_plan is FakeExecutionPlanType.PUBLIC
        assert config.queue_priority is FakeQueuePriorityType.STANDARD
        if self.fail_on_call == self.calls:
            raise RuntimeError("synthetic preparation failure")
        assert b"$0" in file_content
        assert b"qubit[" not in file_content
        self.submitted_bytes.append(file_content)
        job = FakeJob(f"job-{self.calls}", config.name)
        self.jobs.append(job)
        self.outputs[job.id] = {"counts": {"0x0": 3, "0x3": 1}}
        return job

    def download_job_output(self, job):
        return self.outputs[job.id]

    def get_job(self, job_id):
        return next(job for job in self.jobs if job.id == job_id)


class FakeExecutionPlanType(Enum):
    PUBLIC = 'public'
    PRIVATE = 'private'


class FakeQueuePriorityType(Enum):
    STANDARD = 'standard'
    PRIORITY = 'priority'
    INSTANT = 'instant'


class FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def provider_with_scheduler(scheduler: FakeScheduler):
    provider = object.__new__(qpu_provider.OpenQuantumProvider)
    provider.service = SimpleNamespace(scheduler=scheduler)
    provider.backend = lambda *args, **kwargs: SimpleNamespace(num_qubits=108)
    return provider


def test_extract_counts_normalises_hexadecimal_keys():
    counts = qpu_provider._extract_counts(
        {"results": [{"counts": {"0x0": 3, "0x3": 1}}]},
        shots=4,
        width=2,
    )
    assert counts == {"00": 3, "11": 1}


def test_raw_physical_submission_uses_one_job_per_circuit_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(qpu_provider, "_core_imports", lambda: (FakeConfig, FakeExecutionPlanType, FakeQueuePriorityType))
    scheduler = FakeScheduler()
    provider = provider_with_scheduler(scheduler)
    qasm_a = tmp_path / "a.qasm3"
    qasm_b = tmp_path / "b.qasm3"
    qasm_a.write_text(PHYSICAL_QASM, encoding="utf-8")
    qasm_b.write_text(PHYSICAL_QASM.replace("x $0;", "x $1;"), encoding="utf-8")
    output = tmp_path / "run"

    result = provider.submit_bundle(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=[qasm_a, qasm_b],
        labels=["candidate::basis_all0", "candidate::basis_all1"],
        shots=4,
        output=output,
        stage="probes",
    )

    assert scheduler.calls == 2
    assert len(result["results"]) == 2
    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    assert state["status"] == "COMPLETED"
    assert [item["status"] for item in state["circuits"]] == ["COMPLETED", "COMPLETED"]
    assert (output / "raw_jobs" / "000.json").is_file()
    assert (output / "raw_jobs" / "001.json").is_file()


def test_resume_skips_completed_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qpu_provider, "_core_imports", lambda: (FakeConfig, FakeExecutionPlanType, FakeQueuePriorityType))
    qasm_a = tmp_path / "a.qasm3"
    qasm_b = tmp_path / "b.qasm3"
    qasm_a.write_text(PHYSICAL_QASM, encoding="utf-8")
    qasm_b.write_text(PHYSICAL_QASM.replace("x $0;", "x $1;"), encoding="utf-8")
    output = tmp_path / "run"

    first_scheduler = FakeScheduler(fail_on_call=2)
    first_provider = provider_with_scheduler(first_scheduler)
    with pytest.raises(RuntimeError, match="synthetic preparation failure"):
        first_provider.submit_bundle(
            backend_name="rigetti:cepheus-1-108q",
            qasm_files=[qasm_a, qasm_b],
            labels=["a", "b"],
            shots=4,
            output=output,
            stage="probes",
        )

    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    assert state["circuits"][0]["status"] == "COMPLETED"
    assert state["circuits"][1]["status"] == "ERROR"

    second_scheduler = FakeScheduler()
    second_provider = provider_with_scheduler(second_scheduler)
    result = second_provider.submit_bundle(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=[qasm_a, qasm_b],
        labels=["a", "b"],
        shots=4,
        output=output,
        stage="probes",
    )
    assert second_scheduler.calls == 1
    assert len(result["results"]) == 2


def test_import_completed_portal_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qpu_provider, "_core_imports", lambda: (FakeConfig, FakeExecutionPlanType, FakeQueuePriorityType))
    qasm = tmp_path / "a.qasm3"
    qasm.write_text(PHYSICAL_QASM, encoding="utf-8")
    output = tmp_path / "run"
    scheduler = FakeScheduler()
    job = FakeJob("portal-job", "legacy-sampler-job")
    scheduler.jobs.append(job)
    scheduler.outputs[job.id] = {"counts": {"0x0": 4}}
    provider = provider_with_scheduler(scheduler)

    provider.submit_bundle(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=[qasm],
        labels=["9bbe54b1ba83a355::basis_all0"],
        shots=4,
        output=output,
        stage="probes",
        prepare_only=True,
    )
    imported = provider.import_completed_job(
        output=output,
        job_id="portal-job",
        label="9bbe54b1ba83a355::basis_all0",
    )
    assert imported["status"] == "IMPORTED"
    assert imported["remaining"] == 0
    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    assert state["circuits"][0]["imported_from_portal"] is True


def test_core_sdk_explicit_choices_are_converted_to_enums():
    plan = qpu_provider._coerce_sdk_choice(
        'public', FakeExecutionPlanType, field='execution_plan'
    )
    priority = qpu_provider._coerce_sdk_choice(
        'standard', FakeQueuePriorityType, field='queue_priority'
    )
    assert plan is FakeExecutionPlanType.PUBLIC
    assert priority is FakeQueuePriorityType.STANDARD


def test_core_sdk_auto_choice_remains_string():
    assert qpu_provider._coerce_sdk_choice(
        'auto', FakeExecutionPlanType, field='execution_plan'
    ) == 'auto'


def test_core_sdk_invalid_choice_fails_before_network():
    with pytest.raises(ValueError, match='Unsupported execution_plan'):
        qpu_provider._coerce_sdk_choice(
            'public-ish', FakeExecutionPlanType, field='execution_plan'
        )
