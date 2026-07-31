# Public release audit

The GitHub release was constructed from the current scientific source tree, not
from the complete local working directory.

Excluded classes include:

- credential and API-secret files;
- provider job identifiers and submission ledgers;
- account, email, and organization metadata;
- local drive paths and personal operational filenames;
- historical account/tentative/retry records;
- patch installers, patch archives, backup trees, caches, build outputs, and
  compiled local binaries;
- private raw provider artifacts.

Included public evidence is limited to sanitized QASM, aggregate metrics,
scientific source code, tests, current configurations, and the published
manuscript.
