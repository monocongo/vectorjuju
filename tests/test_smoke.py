import vectorjuju


def test_package_imports_with_version():
    assert isinstance(vectorjuju.__version__, str)
    assert vectorjuju.__version__
