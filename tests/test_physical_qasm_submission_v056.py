from pathlib import Path

import pytest

from codegap_qa.qpu_provider import (
    _load_submission_circuit,
    _qasm_addressing,
    _validate_submission_qasm,
)


def test_qasm_addressing_detection_and_bounds():
    assert _qasm_addressing('OPENQASM 3.0;\nqubit[108] q;\nx q[0];\n') == 'register'
    assert _qasm_addressing('OPENQASM 3.0;\nx $66;\n') == 'physical'
    assert _qasm_addressing('OPENQASM 3.0;\nbit[1] c;\n') == 'missing'
    payload = _validate_submission_qasm(
        'OPENQASM 3.0;\nx $66;\n', backend_num_qubits=108
    )
    assert payload['physical_qubits'] == [66]
    with pytest.raises(RuntimeError, match='backend width'):
        _validate_submission_qasm('OPENQASM 3.0;\nx $108;\n', backend_num_qubits=108)
    with pytest.raises(RuntimeError, match='neither a qubit register'):
        _validate_submission_qasm('OPENQASM 3.0;\nbit[1] c;\n', backend_num_qubits=108)


class _PhysicalCircuit:
    def __init__(self, width: int):
        self.num_qubits = width
        self.num_clbits = 48
        self.name = 'probe'
        self.global_phase = 0.0
        self.metadata = {}
        self.data = []
        self.physicalized = False

    def ensure_physical(self, num_qubits=None):
        if num_qubits is not None:
            self.num_qubits = int(num_qubits)
        self.physicalized = True
        return True


class _Qasm3:
    @staticmethod
    def loads(source, num_qubits=None):
        assert '$66' in source
        return _PhysicalCircuit(int(num_qubits))

    @staticmethod
    def dumps(circuit):
        assert circuit.physicalized
        return 'OPENQASM 3.0;\nbit[48] c;\nx $66;\nc[0] = measure $66;\n'


class _UnusedQuantumCircuit:
    pass


def test_load_submission_circuit_preserves_physical_addressing(tmp_path: Path):
    path = tmp_path / 'probe.qasm3'
    path.write_text(
        'OPENQASM 3.0;\nbit[48] c;\nx $66;\nc[0] = measure $66;\n',
        encoding='utf-8',
    )
    circuit, source, diagnostics = _load_submission_circuit(
        path=path,
        qasm3=_Qasm3,
        QuantumCircuit=_UnusedQuantumCircuit,
        backend_num_qubits=108,
    )
    assert circuit.num_qubits == 108
    assert '$66' in source
    assert diagnostics['source_addressing'] == 'physical'
    assert diagnostics['submission_addressing'] == 'physical'
    assert diagnostics['physicalization'] == 'QuantumCircuit.ensure_physical'
    assert diagnostics['backend_num_qubits'] == 108
