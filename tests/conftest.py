def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: builds several Chronos models; deselect with -m 'not slow'"
    )
