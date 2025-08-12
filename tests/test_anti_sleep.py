import sys
import types

from src.ops.anti_sleep_win import AntiSleepGuard, _is_windows


def test_antisleep_noop_on_non_windows(monkeypatch):
    # Force non-Windows behavior
    monkeypatch.setattr("os.name", "posix", raising=False)
    g = AntiSleepGuard(enabled=True)
    assert g.start() is False  # no-op returns False
    assert g.stop() is False


def test_antisleep_calls_windows_api(monkeypatch):
    # Simulate Windows
    monkeypatch.setattr("os.name", "nt", raising=False)

    class DummyKernel:
        def SetThreadExecutionState(self, flags):  # noqa: N802 (Windows API style)
            # Return non-zero truthy
            return 1

    class DummyWindll:
        kernel32 = DummyKernel()

    dummy_ctypes = types.SimpleNamespace(windll=DummyWindll())

    monkeypatch.setitem(sys.modules, "ctypes", dummy_ctypes)

    g = AntiSleepGuard(enabled=True)
    assert g.start() is True
    assert g.stop() is True
