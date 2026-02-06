def test_import_seda():
    import seda
    assert isinstance(seda.__version__, str)

#def test_import_seda():
#    import importlib
#    m = importlib.import_module("seda")
#    assert hasattr(m, "__version__"), "seda should expose __version__"
