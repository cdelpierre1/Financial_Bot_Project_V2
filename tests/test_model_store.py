from pathlib import Path
import pickle

from src.prediction.model_store import ModelStore


def test_model_store_save_load_list(tmp_path, monkeypatch):
    # Settings redirect to tmp dirs
    settings = {
        "paths": {
            "models_trained": str(tmp_path / "trained_models"),
            "models_backup": str(tmp_path / "backup_models"),
        }
    }
    ms = ModelStore(settings=settings)

    # Dummy model
    model = {"coef": [1.0, 2.0], "bias": 0.5}

    # Save
    mp = ms.save("bitcoin", 10, model, metadata={"algo": "dummy"}, do_backup=False)
    assert Path(mp).exists()

    # List
    items = ms.list_models()
    assert any(p.endswith("bitcoin__10m.pkl") for p in items)

    # Load
    m2, meta = ms.load("bitcoin", 10)
    assert m2 == model
    assert meta and meta.get("algo") == "dummy" and meta.get("coin_id") == "bitcoin" and meta.get("horizon_minutes") == 10
