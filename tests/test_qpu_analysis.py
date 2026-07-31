from codegap_qa.qpu_analyze import analyze_probes


def test_probe_analysis_passes_clean_data():
    manifest = {
        'candidate_id': 'x',
        'logical_qubits': 4,
        'circuits': [
            {'label':'basis_all0','type':'basis','expected_pattern':[0,0,0,0]},
            {'label':'basis_all1','type':'basis','expected_pattern':[1,1,1,1]},
            {'label':'basis_even','type':'basis','expected_pattern':[0,1,0,1]},
            {'label':'basis_odd','type':'basis','expected_pattern':[1,0,1,0]},
            {'label':'echo_x1','type':'echo','repetitions':1,'expected_pattern':[0,0,0,0]},
            {'label':'echo_x2','type':'echo','repetitions':2,'expected_pattern':[0,0,0,0]},
            {'label':'echo_x4','type':'echo','repetitions':4,'expected_pattern':[0,0,0,0]},
        ],
    }
    def result(label, key):
        return {'label':label,'counts':{key:1000}}
    raw = {'results':[
        result('basis_all0','0000'),
        result('basis_all1','1111'),
        result('basis_even','1010'),
        result('basis_odd','0101'),
        result('echo_x1','0000'),
        result('echo_x2','0000'),
        result('echo_x4','0000'),
    ]}
    analysis = analyze_probes(
        manifest, raw, science_depth=5,
        minimum_convention_accuracy=0.99,
        maximum_readout_error=0.01,
        maximum_model_mismatch=0.01,
    )
    assert analysis.passed_integrity
    assert analysis.convention == 'qiskit'
    assert analysis.tv_noise_proxy == 0.0
