import json
import types

import src.ops.cli as cli


def test_status_includes_system_metrics(monkeypatch, capsys, tmp_path):
    # Point settings to a temp drive path
    def fake_settings():
        return {
            "paths": {
                "data_parquet": str(tmp_path / "parquet"),
                "runtime": str(tmp_path / "runtime"),
                "base_drive": str(tmp_path.drive) if hasattr(tmp_path, "drive") else None,
            }
        }

    monkeypatch.setattr(cli, "_settings", fake_settings)

    # Avoid touching real torch.cuda
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(cli.sys.modules, "torch", fake_torch)

    cli.status(detail=False)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    sysm = payload.get("system", {})

    assert isinstance(sysm, dict)
    # Basic keys presence
    for k in ("uptime_sec", "ram_used_pct", "disk_free_gb", "disk_total_gb", "gpu_available"):
        assert k in sysm
