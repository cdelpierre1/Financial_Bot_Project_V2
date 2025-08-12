"""
ModelStore — Persistance simple de modèles (pickle) et métadonnées.

- Emplacements pris depuis settings.json: paths.models_trained, paths.models_backup
- Nom standard: {coin_id}__{horizon_minutes}m.pkl (+ .meta.json)
- Sauvegarde avec backup optionnel de l'ancienne version (horodatée)
- Charge le modèle et ses métadonnées si présents
"""
from __future__ import annotations

import os
import json
import pickle
import shutil
import glob
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple, List


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _settings() -> Dict[str, Any]:
    return _load_json(os.path.join(CONFIG_DIR, "settings.json"))


class ModelStore:
    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self.s = settings or _settings()
        paths = self.s.get("paths", {})
        self.root_trained = paths.get("models_trained", os.path.join(ROOT, "models", "trained_models"))
        self.root_backup = paths.get("models_backup", os.path.join(ROOT, "models", "backup_models"))
        os.makedirs(self.root_trained, exist_ok=True)
        os.makedirs(self.root_backup, exist_ok=True)

    @staticmethod
    def _filename(coin_id: str, horizon_minutes: int) -> str:
        return f"{coin_id}__{int(horizon_minutes)}m.pkl"

    def model_path(self, coin_id: str, horizon_minutes: int) -> str:
        return os.path.join(self.root_trained, self._filename(coin_id, horizon_minutes))

    def meta_path(self, coin_id: str, horizon_minutes: int) -> str:
        return os.path.join(self.root_trained, self._filename(coin_id, horizon_minutes) + ".meta.json")

    def backup_model(self, coin_id: str, horizon_minutes: int) -> Optional[str]:
        src = self.model_path(coin_id, horizon_minutes)
        if not os.path.exists(src):
            return None
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dst = os.path.join(self.root_backup, f"{os.path.basename(src)}.{ts}")
        try:
            shutil.copy2(src, dst)
            return dst
        except Exception:
            return None

    def save(self, coin_id: str, horizon_minutes: int, model: Any, metadata: Optional[Dict[str, Any]] = None, do_backup: bool = True) -> str:
        """Sauvegarde le modèle (pickle) et métadonnées JSON. Retourne le chemin du modèle."""
        if do_backup:
            self.backup_model(coin_id, horizon_minutes)
        mp = self.model_path(coin_id, horizon_minutes)
        with open(mp, "wb") as f:
            pickle.dump(model, f)
        meta = metadata.copy() if isinstance(metadata, dict) else {}
        meta.setdefault("coin_id", coin_id)
        meta.setdefault("horizon_minutes", int(horizon_minutes))
        meta.setdefault("saved_at_utc", datetime.now(timezone.utc).isoformat())
        with open(self.meta_path(coin_id, horizon_minutes), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        return mp

    def load(self, coin_id: str, horizon_minutes: int) -> Tuple[Any, Optional[Dict[str, Any]]]:
        mp = self.model_path(coin_id, horizon_minutes)
        if not os.path.exists(mp):
            raise FileNotFoundError(mp)
        with open(mp, "rb") as f:
            model = pickle.load(f)
        meta: Optional[Dict[str, Any]] = None
        mp_meta = self.meta_path(coin_id, horizon_minutes)
        if os.path.exists(mp_meta):
            try:
                with open(mp_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None
        return model, meta

    def list_models(self) -> List[str]:
        if not os.path.isdir(self.root_trained):
            return []
        return [os.path.join(self.root_trained, f) for f in os.listdir(self.root_trained) if f.endswith(".pkl")]

    def cleanup_old_models(self, max_age_days: int = 7, max_models_per_coin_horizon: int = 3) -> Dict[str, Any]:
        """
        Nettoie les anciens modèles selon l'âge et limite le nombre de modèles par coin/horizon.
        
        Stratégie:
        1. Supprimer modèles > max_age_days
        2. Garder max_models_per_coin_horizon meilleurs modèles (par MAE) par coin/horizon
        3. Archiver dans backup avant suppression
        
        Returns: Rapport de nettoyage
        """
        report = {"deleted": [], "kept": [], "errors": [], "stats": {}}
        
        try:
            models = self.list_models()
            if not models:
                report["stats"]["total_models"] = 0
                return report
            
            # Grouper par coin/horizon
            coin_horizon_groups = {}
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            
            for model_path in models:
                try:
                    # Parser le nom: coin__horizonm.pkl
                    filename = os.path.basename(model_path)
                    if not filename.endswith(".pkl") or "__" not in filename:
                        continue
                    
                    parts = filename.replace(".pkl", "").split("__")
                    if len(parts) != 2:
                        continue
                    
                    coin_id = parts[0]
                    horizon_str = parts[1]
                    
                    # Charger les métadonnées pour obtenir l'âge et MAE
                    meta_path = model_path + ".meta.json"
                    model_info = {
                        "path": model_path,
                        "coin_id": coin_id,
                        "horizon": horizon_str,
                        "saved_at": None,
                        "mae": float('inf'),  # Pire MAE par défaut
                        "age_days": float('inf')
                    }
                    
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as f:
                                meta = json.load(f)
                            
                            if "saved_at_utc" in meta:
                                saved_at = datetime.fromisoformat(meta["saved_at_utc"].replace('Z', '+00:00'))
                                model_info["saved_at"] = saved_at
                                model_info["age_days"] = (datetime.now(timezone.utc) - saved_at).days
                            
                            # Récupérer MAE (peut être sous différents noms)
                            mae = meta.get("mae") or meta.get("mae_score") or meta.get("test_mae") or float('inf')
                            model_info["mae"] = float(mae)
                            
                        except Exception as e:
                            report["errors"].append(f"Erreur lecture meta {meta_path}: {e}")
                    
                    # Grouper par coin/horizon
                    key = f"{coin_id}__{horizon_str}"
                    if key not in coin_horizon_groups:
                        coin_horizon_groups[key] = []
                    coin_horizon_groups[key].append(model_info)
                    
                except Exception as e:
                    report["errors"].append(f"Erreur analyse {model_path}: {e}")
            
            # Appliquer les règles de nettoyage
            total_deleted = 0
            total_kept = 0
            
            for group_key, models_in_group in coin_horizon_groups.items():
                # Trier par MAE (meilleur en premier) puis par date (récent en premier)
                models_in_group.sort(key=lambda x: (x["mae"], -x["age_days"]))
                
                for i, model_info in enumerate(models_in_group):
                    should_delete = False
                    reason = ""
                    
                    # Règle 1: Supprimer si trop vieux
                    if model_info["age_days"] > max_age_days:
                        should_delete = True
                        reason = f"trop vieux ({model_info['age_days']} jours)"
                    
                    # Règle 2: Garder seulement les N meilleurs
                    elif i >= max_models_per_coin_horizon:
                        should_delete = True
                        reason = f"dépasse limite ({i+1} > {max_models_per_coin_horizon})"
                    
                    if should_delete:
                        # Backup avant suppression
                        try:
                            model_path = model_info["path"]
                            meta_path = model_path + ".meta.json"
                            
                            # Créer backup avec timestamp
                            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            backup_model = os.path.join(self.root_backup, f"{os.path.basename(model_path)}.{ts}")
                            backup_meta = backup_model + ".meta.json"
                            
                            if os.path.exists(model_path):
                                shutil.copy2(model_path, backup_model)
                                os.remove(model_path)
                            
                            if os.path.exists(meta_path):
                                shutil.copy2(meta_path, backup_meta)
                                os.remove(meta_path)
                            
                            report["deleted"].append({
                                "coin_horizon": group_key,
                                "path": model_path,
                                "reason": reason,
                                "mae": model_info["mae"],
                                "age_days": model_info["age_days"],
                                "backup": backup_model
                            })
                            total_deleted += 1
                            
                        except Exception as e:
                            report["errors"].append(f"Erreur suppression {model_info['path']}: {e}")
                    else:
                        report["kept"].append({
                            "coin_horizon": group_key,
                            "path": model_info["path"],
                            "mae": model_info["mae"],
                            "age_days": model_info["age_days"],
                            "rank": i + 1
                        })
                        total_kept += 1
            
            report["stats"] = {
                "total_models": len(models),
                "total_deleted": total_deleted,
                "total_kept": total_kept,
                "coin_horizon_groups": len(coin_horizon_groups)
            }
            
        except Exception as e:
            report["errors"].append(f"Erreur générale cleanup: {e}")
        
        return report

    def get_model_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les modèles présents."""
        models = self.list_models()
        stats = {
            "total_models": len(models),
            "by_coin": {},
            "by_horizon": {},
            "oldest_model": None,
            "newest_model": None,
            "best_mae_by_coin": {}
        }
        
        oldest_date = None
        newest_date = None
        
        for model_path in models:
            try:
                filename = os.path.basename(model_path)
                if "__" not in filename:
                    continue
                
                parts = filename.replace(".pkl", "").split("__")
                if len(parts) != 2:
                    continue
                
                coin_id = parts[0]
                horizon_str = parts[1]
                
                # Compter par coin et horizon
                stats["by_coin"][coin_id] = stats["by_coin"].get(coin_id, 0) + 1
                stats["by_horizon"][horizon_str] = stats["by_horizon"].get(horizon_str, 0) + 1
                
                # Analyser métadonnées
                meta_path = model_path + ".meta.json"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        
                        if "saved_at_utc" in meta:
                            saved_at = datetime.fromisoformat(meta["saved_at_utc"].replace('Z', '+00:00'))
                            
                            if oldest_date is None or saved_at < oldest_date:
                                oldest_date = saved_at
                                stats["oldest_model"] = {"path": model_path, "date": saved_at.isoformat()}
                            
                            if newest_date is None or saved_at > newest_date:
                                newest_date = saved_at
                                stats["newest_model"] = {"path": model_path, "date": saved_at.isoformat()}
                        
                        # Tracker meilleur MAE par coin
                        mae = meta.get("mae") or meta.get("mae_score") or meta.get("test_mae")
                        if mae is not None:
                            current_best = stats["best_mae_by_coin"].get(coin_id, {}).get("mae", float('inf'))
                            if float(mae) < current_best:
                                stats["best_mae_by_coin"][coin_id] = {
                                    "mae": float(mae),
                                    "horizon": horizon_str,
                                    "path": model_path
                                }
                    
                    except Exception:
                        pass
            
            except Exception:
                continue
        
        return stats
