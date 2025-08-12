"""
Parquet Writer utilitaire

Fonctions principales:
- Écriture partitionnée (date, coin_id) dans src/data/parquet/<dataset>/
- Déduplication intra-batch sur clés (ex: ["coin_id","timeframe","ts_utc_ms"])
- Rétention: suppression des partitions (date=YYYY-MM-DD) plus anciennes que la fenêtre définie
- (Optionnel) Compaction: regroupe les fichiers d'une partition en un seul fichier dédupliqué

Datasets attendus (spec): five_min, hourly, daily, markets, tickers_spread, fx

Prérequis: pandas, pyarrow
"""
from __future__ import annotations

import os
import json
import uuid
import shutil
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")


def _load_settings() -> Dict:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class ParquetWriter:
    def __init__(self, settings: Optional[Dict] = None) -> None:
        self.settings = settings or _load_settings()
        # Résoudre le chemin absolu basé sur la racine du projet (où est settings.json)
        config_dir = os.path.dirname(SETTINGS_PATH)  # src/config
        project_root = os.path.dirname(config_dir)   # src/
        relative_path = self.settings["paths"]["data_parquet"]  # "src/data/parquet"
        # Si le chemin commence par "src/", on l'adapte à la vraie racine
        if relative_path.startswith("src/"):
            self.root = os.path.abspath(os.path.join(project_root, relative_path[4:]))  # enlever "src/" et joindre à src/
        else:
            self.root = os.path.abspath(os.path.join(project_root, relative_path))
        # Retention en jours par timeframe
        self.retention = self.settings.get("parquet", {}).get("retention_days", {})
        self.compression = self.settings.get("parquet", {}).get("compression", "snappy")

    def _dataset_root(self, dataset: str) -> str:
        path = os.path.join(self.root, dataset)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

    @staticmethod
    def _ensure_date_partition(df: pd.DataFrame) -> pd.DataFrame:
        if "ts_utc_ms" not in df.columns:
            raise ValueError("Colonne 'ts_utc_ms' requise pour la partition date")
        # Crée une colonne date=YYYY-MM-DD (UTC)
        d = pd.to_datetime(df["ts_utc_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        if "date" in df.columns:
            df = df.drop(columns=["date"])  # normalise
        df = df.copy()
        df["date"] = d
        return df

    @staticmethod
    def _in_batch_dedup(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        if not keys:
            return df
        # Garde la dernière occurrence par clés
        return df.drop_duplicates(subset=keys, keep="last")

    def write(self,
              dataset: str,
              df: pd.DataFrame,
              dedup_keys: Optional[List[str]] = None,
              partition_cols: Optional[List[str]] = None,
              allow_empty: bool = False) -> int:
        """
        Écrit un DataFrame en Parquet partitionné.

        - dataset: nom du dossier sous src/data/parquet
        - dedup_keys: clés pour dédup intra-batch (ex: ["coin_id","timeframe","ts_utc_ms"])
        - partition_cols: par défaut ["date","coin_id"] si coin_id existe, sinon ["date"]
        - allow_empty: si True, ne lève pas d'erreur pour df vide

        Retourne le nombre de lignes écrites (après dédup intra-batch)
        """
        if df is None or df.empty:
            if allow_empty:
                return 0
            raise ValueError("DataFrame vide fourni à ParquetWriter.write")

        df = self._ensure_date_partition(df)
        if dedup_keys:
            df = self._in_batch_dedup(df, dedup_keys)

        # Partition par date et coin si présent
        if partition_cols is None:
            partition_cols = ["date"] + (["coin_id"] if "coin_id" in df.columns else [])

        # Écriture par groupe de partitions pour éviter collisions
        root = self._dataset_root(dataset)
        written_rows = 0

        # S'assurer des types basiques
        table = pa.Table.from_pandas(df, preserve_index=False)
        # Éclater en groupes de partitions côté pandas pour contrôler le chemin
        pdf = table.to_pandas()  # conversion sûre pour groupby

        group_cols = [c for c in partition_cols if c in pdf.columns]
        if not group_cols:
            group_cols = []

        if group_cols:
            grouped = pdf.groupby(group_cols, dropna=False)
            for keys_vals, part_df in grouped:
                part_path = root
                if not isinstance(keys_vals, tuple):
                    keys_vals = (keys_vals,)
                for k, v in zip(group_cols, keys_vals):
                    part_path = os.path.join(part_path, f"{k}={v}")
                os.makedirs(part_path, exist_ok=True)
                file_name = f"part-{uuid.uuid4().hex}.parquet"
                file_path = os.path.join(part_path, file_name)
                pq.write_table(pa.Table.from_pandas(part_df, preserve_index=False),
                               file_path,
                               compression=self.compression)
                written_rows += len(part_df)
        else:
            os.makedirs(root, exist_ok=True)
            file_name = f"part-{uuid.uuid4().hex}.parquet"
            file_path = os.path.join(root, file_name)
            pq.write_table(table, file_path, compression=self.compression)
            written_rows = df.shape[0]

        return written_rows

    def append_data(self, dataset: str, df: pd.DataFrame, dedup_keys: Optional[List[str]] = None) -> int:
        """
        Alias pour `write` avec des paramètres par défaut pour l'ajout de données.
        """
        return self.write(dataset, df, dedup_keys=dedup_keys)

    def enforce_retention(self, dataset: str, timeframe: Optional[str] = None) -> int:
        """
        Supprime les partitions trop anciennes selon settings.parquet.retention_days.
        - timeframe: clé pour chercher le nombre de jours (five_min|hourly|daily), sinon aucune suppression.
        Retourne le nombre de dossiers partition supprimés.
        """
        if timeframe is None:
            return 0
        days = int(self.retention.get(timeframe, 0))
        if days <= 0:
            return 0

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        root = self._dataset_root(dataset)
        removed = 0

        # On parcourt les partitions date=YYYY-MM-DD à la racine du dataset
        if not os.path.isdir(root):
            return 0
        for entry in os.listdir(root):
            if not entry.startswith("date="):
                # Ignorer autres répertoires (ex: métadonnées)
                continue
            try:
                part_date = datetime.strptime(entry.split("=", 1)[1], "%Y-%m-%d").date()
            except Exception:
                continue
            if part_date < cutoff:
                try:
                    shutil.rmtree(os.path.join(root, entry), ignore_errors=True)
                    removed += 1
                except Exception:
                    # On laisse l'appelant logger l'erreur si besoin
                    pass
        return removed

    def compact_partition(self, dataset: str, date_str: str, coin_id: Optional[str] = None,
                           dedup_keys: Optional[List[str]] = None) -> Optional[str]:
        """
        Compacte une partition (date=YYYY-MM-DD[/coin_id=<id>]) en un seul fichier dédupliqué.
        Retourne le chemin du fichier compacté si succès, sinon None.
        """
        root = self._dataset_root(dataset)
        part_path = os.path.join(root, f"date={date_str}")
        if coin_id:
            part_path = os.path.join(part_path, f"coin_id={coin_id}")
        if not os.path.isdir(part_path):
            return None

        # Lire tous les fichiers parquet de la partition
        files = [os.path.join(part_path, f) for f in os.listdir(part_path) if f.endswith(".parquet")]
        if not files:
            return None
        frames = []
        for fp in files:
            try:
                frames.append(pq.read_table(fp).to_pandas())
            except Exception:
                continue
        if not frames:
            return None
        pdf = pd.concat(frames, ignore_index=True)
        if dedup_keys:
            pdf = pdf.drop_duplicates(subset=dedup_keys, keep="last")

        # Écrire un seul fichier et supprimer les anciens
        tmp_name = f"compact-{uuid.uuid4().hex}.parquet"
        tmp_path = os.path.join(part_path, tmp_name)
        pq.write_table(pa.Table.from_pandas(pdf, preserve_index=False), tmp_path, compression=self.compression)

        # Nettoyage anciens fichiers
        for fp in files:
            try:
                os.remove(fp)
            except Exception:
                pass
        return tmp_path


__all__ = ["ParquetWriter"]
"""Placeholder parquet_writer — SPEC UNIQUEMENT."""
