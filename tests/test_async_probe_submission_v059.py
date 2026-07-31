from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace

from codegap_qa import qpu_provider


PHYSICAL_QASM = """OPENQASM 3.0;
bit[2] c;
x $0;
c[0] = measure $0;
c[1] = measure $1;
"""


class FakeExecutionPlanType(Enum):
    PUBLIC = "public"
    PRIVATE = "private"


class FakeQueuePriorityType(Enum):
    STANDARD = "standard"
    PRIORITY = "priority"


class FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakePreparationCreate:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeJobCreate:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeJob:
    def __init__(self, job_id: str, name: str, status: str = "Queued"):
        self.id = job_id
        self.name = name
        self.status = status
        self.output_data_url = None


class FakeJobList:
    def __init__(self, jobs):
        self.jobs = list(jobs)


class FakeScheduler:
    def __init__(self):
        self.jobs: list[FakeJob] = []
        self.outputs: dict[str, dict] = {}
        self.uploads = 0
        self.preparations = 0
        self.creates = 0

    def list_jobs(self, status=None):
        jobs = self.jobs
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        return FakeJobList(jobs)

    def upload_job_input(self, *, file_content):
        assert b"$0" in file_content or b"$1" in file_content
        self.uploads += 1
        return f"upload-{self.uploads}"

    def prepare_job(self, preparation):
        self.preparations += 1
        return SimpleNamespace(id=f"prep-{self.preparations}")

    def get_preparation_result(self, preparation_id):
        return SimpleNamespace(status="Completed", quote=[object()])

    def _choose_plan_and_priority(self, result, config):
        assert config.execution_plan is FakeExecutionPlanType.PUBLIC
        assert config.queue_priority is FakeQueuePriorityType.STANDARD
        return "plan-public", "queue-standard"

    def create_job(self, create):
        self.creates += 1
        job = FakeJob(f"job-{self.creates}", f"created-{self.creates}")
        self.jobs.append(job)
        self.outputs[job.id] = {"counts": {"0x0": 3, "0x3": 1}}
        return job

    def get_job(self, job_id):
        return next(job for job in self.jobs if job.id == job_id)

    def download_job_output(self, job):
        return self.outputs[job.id]


class FakeManagement:
    def list_user_organizations(self):
        return SimpleNamespace(organizations=[SimpleNamespace(id="org-1")])


def provider_with_scheduler(scheduler: FakeScheduler):
    provider = object.__new__(qpu_provider.OpenQuantumProvider)
    provider.service = SimpleNamespace(
        scheduler=scheduler,
        management=FakeManagement(),
    )
    provider.backend = lambda *args, **kwargs: SimpleNamespace(num_qubits=108)
    return provider


def async_imports():
    return (
        FakeConfig,
        FakeExecutionPlanType,
        FakeQueuePriorityType,
        FakePreparationCreate,
        FakeJobCreate,
    )


def make_qasm_files(tmp_path: Path):
    first = tmp_path / "a.qasm3"
    second = tmp_path / "b.qasm3"
    third = tmp_path / "c.qasm3"
    first.write_text(PHYSICAL_QASM, encoding="utf-8")
    second.write_text(PHYSICAL_QASM.replace("x $0;", "x $1;"), encoding="utf-8")
    third.write_text(PHYSICAL_QASM, encoding="utf-8")
    return [first, second, third]


def test_attach_running_job_then_submit_remaining_without_waiting(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(qpu_provider, "_core_async_imports", async_imports)
    scheduler = FakeScheduler()
    provider = provider_with_scheduler(scheduler)
    qasm_files = make_qasm_files(tmp_path)
    labels = ["candidate::basis_all0", "candidate::basis_all1", "candidate::basis_even"]
    output = tmp_path / "run"

    # Initialize the v0.5.7 state, with the first job already recovered.
    provider.submit_bundle(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=qasm_files,
        labels=labels,
        shots=4,
        output=output,
        stage="probes",
        prepare_only=True,
    )
    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    state["circuits"][0].update(
        {
            "status": "COMPLETED",
            "job_id": "legacy-completed",
            "shots_received": 4,
            "distinct_outcomes": 1,
            "counts": {"00": 4},
        }
    )
    (output / "submission.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    running = FakeJob("running-job", state["circuits"][1]["job_name"], "Running")
    scheduler.jobs.append(running)
    scheduler.outputs[running.id] = {"counts": {"0x0": 4}}
    attached = provider.attach_async_job(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=qasm_files,
        labels=labels,
        shots=4,
        output=output,
        stage="probes",
        execution_plan="public",
        queue_priority="standard",
        job_id="running-job",
        label=None,
    )
    assert attached["label"] == "candidate::basis_all1"

    result = provider.submit_bundle_async(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=qasm_files,
        labels=labels,
        shots=4,
        output=output,
        stage="probes",
        execution_plan="public",
        queue_priority="standard",
        max_active=10,
    )
    assert result["status"] == "ALL_JOBS_SUBMITTED"
    assert scheduler.creates == 1
    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    assert state["circuits"][1]["job_id"] == "running-job"
    assert state["circuits"][2]["job_id"] == "job-1"
    assert not (output / "raw_counts.json").exists()


def test_collect_after_restart_downloads_all_results(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(qpu_provider, "_core_async_imports", async_imports)
    scheduler = FakeScheduler()
    provider = provider_with_scheduler(scheduler)
    qasm_files = make_qasm_files(tmp_path)[:2]
    labels = ["candidate::basis_all0", "candidate::basis_all1"]
    output = tmp_path / "run"

    submitted = provider.submit_bundle_async(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=qasm_files,
        labels=labels,
        shots=4,
        output=output,
        stage="probes",
        execution_plan="public",
        queue_priority="standard",
        max_active=10,
    )
    assert submitted["status"] == "ALL_JOBS_SUBMITTED"
    for job in scheduler.jobs:
        job.status = "Completed"
        job.output_data_url = "https://example.invalid/result"

    # A new provider instance represents a restarted PC/process.
    restarted = provider_with_scheduler(scheduler)
    collected = restarted.collect_async_bundle(output=output, wait=False)
    assert collected["status"] == "RESULTS_COLLECTED"
    payload = json.loads((output / "raw_counts.json").read_text(encoding="utf-8"))
    assert len(payload["results"]) == 2
    assert all(sum(row["counts"].values()) == 4 for row in payload["results"])


def test_existing_job_name_is_recovered_after_interruption(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(qpu_provider, "_core_async_imports", async_imports)
    scheduler = FakeScheduler()
    provider = provider_with_scheduler(scheduler)
    qasm_files = make_qasm_files(tmp_path)[:1]
    labels = ["candidate::basis_all0"]
    output = tmp_path / "run"

    provider.submit_bundle(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=qasm_files,
        labels=labels,
        shots=4,
        output=output,
        stage="probes",
        prepare_only=True,
    )
    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    portal_job = FakeJob("already-created", state["circuits"][0]["job_name"], "Queued")
    scheduler.jobs.append(portal_job)

    result = provider.submit_bundle_async(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=qasm_files,
        labels=labels,
        shots=4,
        output=output,
        stage="probes",
        execution_plan="public",
        queue_priority="standard",
        max_active=10,
    )
    assert result["status"] == "ALL_JOBS_SUBMITTED"
    assert scheduler.creates == 0
    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    assert state["circuits"][0]["job_id"] == "already-created"


def test_collect_specific_job_while_other_probes_have_no_job_ids(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(qpu_provider, "_core_async_imports", async_imports)
    scheduler = FakeScheduler()
    provider = provider_with_scheduler(scheduler)
    qasm_files = make_qasm_files(tmp_path)
    labels = ["candidate::basis_all0", "candidate::basis_all1", "candidate::basis_even"]
    output = tmp_path / "run"

    provider.submit_bundle(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=qasm_files,
        labels=labels,
        shots=4,
        output=output,
        stage="probes",
        prepare_only=True,
    )
    initial = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    job = FakeJob("public-test-job", initial["circuits"][1]["job_name"], "Completed")
    job.output_data_url = "https://example.invalid/result"
    scheduler.jobs.append(job)
    scheduler.outputs[job.id] = {"counts": {"0x0": 2, "0x3": 2}}
    provider.attach_async_job(
        backend_name="rigetti:cepheus-1-108q",
        qasm_files=qasm_files,
        labels=labels,
        shots=4,
        output=output,
        stage="probes",
        execution_plan="public",
        queue_priority="standard",
        job_id=job.id,
        label=labels[1],
    )
    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    state["circuits"][0].update(
        {
            "status": "COMPLETED",
            "job_id": "legacy-completed",
            "shots_received": 4,
            "distinct_outcomes": 1,
            "counts": {"00": 4},
        }
    )
    (output / "submission.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    result = provider.collect_async_bundle(
        output=output,
        wait=False,
        job_ids=[job.id],
    )
    assert result["status"] == "PARTIAL_RESULTS_COLLECTED"
    assert result["completed"] == 2
    assert result["remaining"] == 1
    state = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    assert state["circuits"][1]["status"] == "COMPLETED"
    assert state["circuits"][2]["job_id"] is None
    assert not (output / "raw_counts.json").exists()
