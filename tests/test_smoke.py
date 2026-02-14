def test_smoke_import():
    # If this fails, your app has import-time errors
    import importlib
    importlib.import_module("app")
