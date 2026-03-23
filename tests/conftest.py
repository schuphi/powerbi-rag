"""Pytest configuration for local src-layout imports."""

import asyncio
import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def pytest_pyfunc_call(pyfuncitem):
    """Run async tests without requiring pytest-asyncio."""
    test_function = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_function):
        kwargs = {
            name: pyfuncitem.funcargs[name]
            for name in inspect.signature(test_function).parameters
        }
        asyncio.run(test_function(**kwargs))
        return True
    return None
