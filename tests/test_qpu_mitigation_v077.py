from __future__ import annotations

import numpy as np

from codegap_qa.qpu_mitigation import (
    assignment_matrix,
    inverse_observable_coefficients,
)


def main() -> None:
    p01 = [0.03, 0.02]
    p10 = [0.08, 0.04]
    matrix = np.kron(
        assignment_matrix(p01[0], p10[0]),
        assignment_matrix(p01[1], p10[1]),
    )
    true_distribution = np.asarray([0.45, 0.10, 0.15, 0.30], dtype=float)
    measured_distribution = matrix @ true_distribution
    coefficients = inverse_observable_coefficients(p01, p10)
    recovered = float(coefficients @ measured_distribution)
    expected = float(np.asarray([1.0, -1.0, -1.0, 1.0]) @ true_distribution)
    if abs(recovered - expected) > 1.0e-12:
        raise RuntimeError(
            f"Readout inversion failed: {recovered} != {expected}"
        )
    print("V077 MITIGATION SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
