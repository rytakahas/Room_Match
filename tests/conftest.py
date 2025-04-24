import pytest

def pytest_collection_modifyitems(items):
    for item in items:
        try:
            content = item.fspath.read_text(encoding="utf-8")
            if "127.0.0.1" in content:
                item.add_marker(pytest.mark.skip(reason="Test requires live server at 127.0.0.1"))
        except Exception as e:
            print(f"Warning: Could not read {item.fspath}: {e}")

