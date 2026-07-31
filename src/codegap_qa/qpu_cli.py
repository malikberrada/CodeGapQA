from __future__ import annotations

import argparse
import json
from pathlib import Path

from .qpu_workflow import (
    capture_qpu_target_snapshot,
    certify_science,
    prepare_qpu_campaign,
    select_from_probes,
    submit_probes,
    attach_probe_job,
    submit_probes_async,
    collect_probes_async,
    import_probe_job,
    list_qpu_jobs,
    submit_science,
)


def add_qpu_commands(commands: argparse._SubParsersAction) -> None:
    snapshot = commands.add_parser('qpu-snapshot')
    snapshot.add_argument('--credentials', type=Path)
    snapshot.add_argument('--backend', required=True)
    snapshot.add_argument('--out', type=Path, required=True)
    snapshot.add_argument('--execution-plan', choices=('public', 'private'), default='public')
    snapshot.add_argument('--queue-priority', default='standard')

    prepare = commands.add_parser('qpu-prepare')
    prepare.add_argument('--artifact', type=Path, required=True)
    prepare.add_argument('--credentials', type=Path)
    prepare.add_argument('--backend', required=True)
    prepare.add_argument('--out', type=Path, required=True)
    prepare.add_argument('--max-candidates', type=int, default=8)
    prepare.add_argument('--max-layouts', type=int, default=512)
    prepare.add_argument('--execution-plan', choices=('public', 'private'), default='public')
    prepare.add_argument('--queue-priority', default='standard')

    probes = commands.add_parser('qpu-run-probes')
    probes.add_argument('--campaign', type=Path, required=True)
    probes.add_argument('--credentials', type=Path)
    probes.add_argument('--shots', type=int, required=True)
    probes.add_argument('--out', type=Path, required=True)

    attach_probe = commands.add_parser('qpu-attach-probe-job')
    attach_probe.add_argument('--campaign', type=Path, required=True)
    attach_probe.add_argument('--credentials', type=Path)
    attach_probe.add_argument('--shots', type=int, required=True)
    attach_probe.add_argument('--out', type=Path, required=True)
    attach_probe.add_argument('--job-id', required=True)
    attach_probe.add_argument('--label', default=None)

    submit_probes_async_parser = commands.add_parser('qpu-submit-probes')
    submit_probes_async_parser.add_argument('--campaign', type=Path, required=True)
    submit_probes_async_parser.add_argument('--credentials', type=Path)
    submit_probes_async_parser.add_argument('--shots', type=int, required=True)
    submit_probes_async_parser.add_argument('--out', type=Path, required=True)
    submit_probes_async_parser.add_argument('--max-active', type=int, default=10)
    submit_probes_async_parser.add_argument('--slot-poll-seconds', type=int, default=15)
    submit_probes_async_parser.add_argument('--slot-timeout-seconds', type=int, default=86400)

    collect_probes = commands.add_parser('qpu-collect-probes')
    collect_probes.add_argument('--campaign', type=Path, required=True)
    collect_probes.add_argument('--credentials', type=Path)
    collect_probes.add_argument('--out', type=Path, required=True)
    collect_probes.add_argument('--wait', action='store_true')
    collect_probes.add_argument('--poll-seconds', type=int, default=20)
    collect_probes.add_argument('--timeout-seconds', type=int, default=86400)
    collect_probes.add_argument(
        '--job-id',
        action='append',
        dest='job_ids',
        default=None,
        help='Collect only the specified Open Quantum job ID. Repeat for multiple jobs.',
    )

    import_probe = commands.add_parser('qpu-import-probe-job')
    import_probe.add_argument('--campaign', type=Path, required=True)
    import_probe.add_argument('--credentials', type=Path)
    import_probe.add_argument('--shots', type=int, required=True)
    import_probe.add_argument('--out', type=Path, required=True)
    import_probe.add_argument('--job-id', required=True)
    import_probe.add_argument('--label', required=True)

    jobs = commands.add_parser('qpu-list-jobs')
    jobs.add_argument('--credentials', type=Path)
    jobs.add_argument('--status', default=None)
    jobs.add_argument('--limit', type=int, default=25)

    select = commands.add_parser('qpu-select')
    select.add_argument('--campaign', type=Path, required=True)
    select.add_argument('--probe-run', type=Path, required=True)
    select.add_argument('--out', type=Path, required=True)

    submit = commands.add_parser('qpu-run')
    submit.add_argument('--selected', type=Path, required=True)
    submit.add_argument('--campaign', type=Path, required=True)
    submit.add_argument('--credentials', type=Path)
    submit.add_argument('--stage', choices=('diagnostic', 'final'), required=True)
    submit.add_argument('--shots', type=int, required=True)
    submit.add_argument('--out', type=Path, required=True)

    certify = commands.add_parser('qpu-certify')
    certify.add_argument('--selected', type=Path, required=True)
    certify.add_argument('--run', type=Path, required=True)
    certify.add_argument('--stage', choices=('diagnostic', 'final'), required=True)
    certify.add_argument('--out', type=Path, required=True)


def handle_qpu_command(args: argparse.Namespace) -> int | None:
    if args.command == 'qpu-snapshot':
        result = capture_qpu_target_snapshot(
            credentials=args.credentials,
            backend_name=args.backend,
            output=args.out,
            execution_plan=args.execution_plan,
            queue_priority=args.queue_priority,
        )
    elif args.command == 'qpu-prepare':
        result = prepare_qpu_campaign(
            artifact_root=args.artifact,
            credentials=args.credentials,
            backend_name=args.backend,
            output=args.out,
            max_candidates=args.max_candidates,
            max_layouts=args.max_layouts,
            execution_plan=args.execution_plan,
            queue_priority=args.queue_priority,
        )
    elif args.command == 'qpu-run-probes':
        result = submit_probes(
            campaign_root=args.campaign,
            credentials=args.credentials,
            shots=args.shots,
            output=args.out,
        )
    elif args.command == 'qpu-attach-probe-job':
        result = attach_probe_job(
            campaign_root=args.campaign,
            credentials=args.credentials,
            shots=args.shots,
            output=args.out,
            job_id=args.job_id,
            label=args.label,
        )
    elif args.command == 'qpu-submit-probes':
        result = submit_probes_async(
            campaign_root=args.campaign,
            credentials=args.credentials,
            shots=args.shots,
            output=args.out,
            max_active=args.max_active,
            slot_poll_seconds=args.slot_poll_seconds,
            slot_timeout_seconds=args.slot_timeout_seconds,
        )
    elif args.command == 'qpu-collect-probes':
        result = collect_probes_async(
            campaign_root=args.campaign,
            credentials=args.credentials,
            output=args.out,
            wait=args.wait,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
            job_ids=args.job_ids,
        )
    elif args.command == 'qpu-import-probe-job':
        result = import_probe_job(
            campaign_root=args.campaign,
            credentials=args.credentials,
            shots=args.shots,
            output=args.out,
            job_id=args.job_id,
            label=args.label,
        )
    elif args.command == 'qpu-list-jobs':
        result = list_qpu_jobs(
            credentials=args.credentials,
            status=args.status,
            limit=args.limit,
        )
    elif args.command == 'qpu-select':
        result = select_from_probes(
            campaign_root=args.campaign,
            probe_run=args.probe_run,
            output=args.out,
        )
    elif args.command == 'qpu-run':
        result = submit_science(
            selected_root=args.selected,
            campaign_root=args.campaign,
            credentials=args.credentials,
            stage=args.stage,
            shots=args.shots,
            output=args.out,
        )
    elif args.command == 'qpu-certify':
        result = certify_science(
            selected_root=args.selected,
            science_run=args.run,
            stage=args.stage,
            output=args.out,
        )
    else:
        return None
    print(json.dumps(result, indent=2))
    status = result.get('status')
    passed = result.get('pass')
    return 0 if status not in {'STOP_NO_QPU_ROBUST_CANDIDATE'} and passed is not False else 2
