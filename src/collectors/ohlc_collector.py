"""
OHLC Collector — CoinGecko /coins/{id}/ohlc

— Récupère des chandelles OHLC pour un coin et une fenêtre (days)
— Mappe vers schéma daily: [ts_utc_ms, coin_id, o, h, l, c, volume, agg_method]
— Écrit dataset=daily via ParquetWriter
"""

from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from datetime import datetime, timezone
from dotenv import dotenv_values

from storage.cache_store import CacheStore
from storage.parquet_writer import ParquetWriter
from ops.api_usage import api_acquire, api_register


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class _KeyRotator:
    def __init__(self, env_var: str = "COINGECKO_API_KEYS") -> None:
        keys_env = os.environ.get(env_var)
        keys: List[str] = []
        if keys_env:
            keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        if not keys:
            env_path = os.path.join(CONFIG_DIR, "secrets", ".env.local")
            if os.path.exists(env_path):
                values = dotenv_values(env_path)
                v = values.get(env_var)
                if v:
                    keys = [k.strip() for k in v.split(",") if k.strip()]
        self._keys = keys
        self._idx = 0

    def next(self) -> Optional[str]:
        if not self._keys:
            return None
        k = self._keys[self._idx % len(self._keys)]
        self._idx += 1
        return k


class OhlcCollector:
    def __init__(self, cache: Optional[CacheStore] = None, writer: Optional[ParquetWriter] = None) -> None:
        self.api_config = _load_json(os.path.join(CONFIG_DIR, "api_config.json"))
        self.coins_config = _load_json(os.path.join(CONFIG_DIR, "coins.json"))
        ttl = 3600
        self.cache = cache or CacheStore(default_ttl_s=ttl)
        self.writer = writer or ParquetWriter()
        self._rotator = _KeyRotator(self.api_config["coingecko"]["auth"]["env_keys_var"])

    def _client(self) -> httpx.Client:
        timeouts = self.api_config["coingecko"].get("timeouts_sec", {"connect": 5, "read": 10})
        connect = float(timeouts.get("connect", 5))
        read = float(timeouts.get("read", 10))
        write = float(timeouts.get("write", read))
        pool = float(timeouts.get("pool", read))
        headers = {"User-Agent": self.api_config["coingecko"].get("user_agent", "fb2/1.0")}
        return httpx.Client(timeout=httpx.Timeout(connect=connect, read=read, write=write, pool=pool), headers=headers)

    def _endpoint(self, coin_id: str) -> str:
        cg = self.api_config["coingecko"]
        return cg["base_url"] + cg["endpoints"]["ohlc"].format(id=coin_id)

    def _params(self, days: int) -> Dict[str, Any]:
        cg = self.api_config["coingecko"]
        vs = cg["default_params"].get("vs_currency", "usd")
        return {"vs_currency": vs, "days": int(days)}

    def fetch_ohlc(self, coin_id: str, days: int = 1) -> Tuple[List[List[float]], int]:
        # Acquire rate limiter slot
        api_acquire("ohlc")
        
        cg = self.api_config["coingecko"]
        url = self._endpoint(coin_id)
        params = self._params(days)
        
        # Retry avec rotation des clés sur 429
        max_retries = len(self._rotator._keys) if self._rotator._keys else 1
        for attempt in range(max_retries):
            headers: Dict[str, str] = {}
            api_key = self._rotator.next()
            if api_key:
                headers[cg["auth"]["header_name"]] = api_key
            
            with self._client() as client:
                r = client.get(url, params=params, headers=headers)
                status = r.status_code
                
                # Register API call result
                api_register("ohlc", status)
                
                if status == 429 and attempt < max_retries - 1:
                    # Rate limit avec cette clé, essayer la suivante
                    print(f"⚠️ Rate limit sur clé {attempt+1}, rotation vers clé {attempt+2}...")
                    time.sleep(1)  # Pause courte avant nouvelle clé
                    continue
                    
                if status >= 500 or status == 429:
                    r.raise_for_status()
                if status >= 400:
                    r.raise_for_status()
                    
                data = r.json()  # [[timestamp, open, high, low, close], ...]
                return data, status
        
        # Si on arrive ici, toutes les clés sont en rate limit
        raise Exception(f"Toutes les clés API sont en rate limit pour {coin_id}")

    def get_ohlc(self, coin_id: str, days: int = 1) -> Dict[str, Any]:
        ttl_s = 3600
        key_parts = {"endpoint": "ohlc", "id": coin_id, "days": int(days), "vs": "usd"}

        def fetcher():
            data, status = self.fetch_ohlc(coin_id, days)
            ts = _now_ms()
            return {"ts_utc_ms": ts, "id": coin_id, "days": int(days), "data": data}, status

        return self.cache.get_or_fetch(key_parts, fetcher, ttl_s=ttl_s, attempts=self.api_config["coingecko"]["retries"].get("max_attempts", 3))

    @staticmethod
    def to_dataframe(coin_id: str, payload: Dict[str, Any]) -> pd.DataFrame:
        data = payload.get("data", [])
        rows: List[Dict[str, Any]] = []
        for item in data:
            try:
                ts_ms = int(item[0])
                o = float(item[1])
                h = float(item[2])
                l = float(item[3])
                c = float(item[4])
                rows.append({
                    "ts_utc_ms": ts_ms,
                    "coin_id": coin_id,
                    "o": o,
                    "h": h,
                    "l": l,
                    "c": c,
                    "volume": None,
                    "agg_method": "coingecko_ohlc",
                })
            except Exception:
                continue
        return pd.DataFrame(rows, columns=["ts_utc_ms","coin_id","o","h","l","c","volume","agg_method"])

    def collect(self, coin_id: str, days: int = 1, write: bool = True) -> Tuple[pd.DataFrame, int]:
        payload = self.get_ohlc(coin_id, days)
        df = self.to_dataframe(coin_id, payload)
        written = 0
        if write and not df.empty:
            written = self.writer.write("daily", df, dedup_keys=["coin_id","ts_utc_ms"], partition_cols=None)
        return df, written

    def collect_daily_batch(self, coin_ids: List[str], days: int = 1) -> Tuple[pd.DataFrame, int]:
        """Collecte OHLC quotidien pour plusieurs coins en batch"""
        all_dfs = []
        total_written = 0
        
        for coin_id in coin_ids:
            try:
                df, written = self.collect(coin_id, days, write=True)
                all_dfs.append(df)
                total_written += written
            except Exception as e:
                print(f"  ❌ Erreur {coin_id}: {e}")
                continue
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            return combined_df, total_written
        return pd.DataFrame(), 0


def main() -> None:
    oc = OhlcCollector()
    df, written = oc.collect("bitcoin", days=1, write=False)
    print(f"rows={len(df)} written={written}")


if __name__ == "__main__":
    main()
