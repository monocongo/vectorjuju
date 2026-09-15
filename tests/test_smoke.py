import vectorjuju


def test_package_imports_with_version():
    assert isinstance(vectorjuju.__version__, str)
    assert vectorjuju.__version__


def test_tracing_api_is_exported():
    from vectorjuju import Run, trace_runs, tracing

    assert trace_runs is tracing.trace_runs
    assert Run is tracing.Run
