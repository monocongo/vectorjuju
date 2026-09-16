import vectorjuju


def test_package_imports_with_version():
    assert isinstance(vectorjuju.__version__, str)
    assert vectorjuju.__version__


def test_tracing_api_is_exported():
    from vectorjuju import Run, trace_runs, tracing

    assert trace_runs is tracing.trace_runs
    assert Run is tracing.Run


def test_calibration_api_is_exported():
    from vectorjuju import Scale, ScaleCalibrationError, calibrate, calibrate_scale

    assert calibrate_scale is calibrate.calibrate_scale
    assert Scale is calibrate.Scale
    assert ScaleCalibrationError is calibrate.ScaleCalibrationError


def test_writer_api_is_exported():
    from vectorjuju import export, write_outputs

    assert write_outputs is export.write_outputs
