"""
Range Collector — CoinGecko market_chart/range (backfill 5m)

— Récupère une série de prix/volumes entre deux timestamps (ms) pour un coin donné.
— Utilise /coins/{id}/market_chart/range avec vs_currency=usd et timestamps UNIX en secondes.
— Transforme vers schéma five_min: [ts_utc_ms, coin_id, o, h, l, c, volume, agg_method]
— Écrit en Parquet dataset=five_min via ParquetWriter (dédup par coin_id, ts_utc_ms).
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import httpx
import time
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


class RangeCollector:
    def __init__(self, cache: Optional[CacheStore] = None, writer: Optional[ParquetWriter] = None) -> None:
        self.api_config = _load_json(os.path.join(CONFIG_DIR, "api_config.json"))
        self.coins_config = _load_json(os.path.join(CONFIG_DIR, "coins.json"))
        # TTL court par défaut; chaque intervalle unique a sa clé cache
        ttl = int(self.api_config["coingecko"]["ttl_sec"].get("market_chart", 300))
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
        return cg["base_url"] + cg["endpoints"]["market_chart_range"].format(id=coin_id)
    
    def _endpoint_market_chart(self, coin_id: str) -> str:
        """Endpoint market_chart pour données historiques complètes (depuis 2013)"""
        cg = self.api_config["coingecko"]
        base_url = cg["base_url"]
        return f"{base_url}/coins/{coin_id}/market_chart"

    def _params(self, frm_ms: int, to_ms: int) -> Dict[str, Any]:
        cg = self.api_config["coingecko"]
        vs = cg["default_params"].get("vs_currency", "usd")
        return {"vs_currency": vs, "from": int(frm_ms // 1000), "to": int(to_ms // 1000)}
    
    def _params_market_chart(self, days: str = "max") -> Dict[str, Any]:
        """Paramètres pour market_chart avec days=max (toutes données depuis 2013)"""
        cg = self.api_config["coingecko"]
        vs = cg["default_params"].get("vs_currency", "usd")
        return {"vs_currency": vs, "days": days}

    def fetch_range(self, coin_id: str, frm_ms: int, to_ms: int) -> Tuple[Dict[str, Any], int]:
        # Acquire rate limiter slot
        api_acquire("market_chart_range")
        
        cg = self.api_config["coingecko"]
        url = self._endpoint(coin_id)
        params = self._params(frm_ms, to_ms)
        
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
                api_register("market_chart_range", status)
                
                if status == 429 and attempt < max_retries - 1:
                    # Rate limit avec cette clé, essayer la suivante
                    print(f"⚠️ Rate limit sur clé {attempt+1}, rotation vers clé {attempt+2}...")
                    time.sleep(1)  # Pause courte avant nouvelle clé
                    continue
                    
                if status >= 500 or status == 429:
                    r.raise_for_status()
                if status >= 400:
                    r.raise_for_status()
                    
                data = r.json()
                return data, status
        
        # Si on arrive ici, toutes les clés sont en rate limit
        raise Exception(f"Toutes les clés API sont en rate limit pour {coin_id}")
    
    def fetch_historical_all(self, coin_id: str) -> Tuple[Dict[str, Any], int]:
        """Récupère TOUTES les données historiques avec market_chart?days=max (depuis 2013)"""
        # Acquire rate limiter slot
        api_acquire("market_chart_historical")
        
        cg = self.api_config["coingecko"]
        url = self._endpoint_market_chart(coin_id)
        params = self._params_market_chart("max")
        
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
                api_register("market_chart_historical", status)
                
                if status == 429 and attempt < max_retries - 1:
                    # Rate limit avec cette clé, essayer la suivante
                    print(f"⚠️ Rate limit sur clé {attempt+1}, rotation vers clé {attempt+2}...")
                    time.sleep(8)  # Pause ULTRA-SÉCURISÉE avant nouvelle clé
                    continue
                    
                if status >= 500 or status == 429:
                    r.raise_for_status()
                if status >= 400:
                    r.raise_for_status()
                    
                data = r.json()
                return data, status
        
        # Si on arrive ici, toutes les clés sont en rate limit
        raise Exception(f"Toutes les clés API sont en rate limit pour {coin_id}")

    def get_range(self, coin_id: str, frm_ms: int, to_ms: int) -> Dict[str, Any]:
        ttl_s = int(self.api_config["coingecko"]["ttl_sec"].get("market_chart", 300))
        key_parts = {"endpoint": "market_chart_range", "id": coin_id, "from": int(frm_ms // 1000), "to": int(to_ms // 1000), "vs": "usd"}

        def fetcher():
            data, status = self.fetch_range(coin_id, frm_ms, to_ms)
            ts = _now_ms()
            return {"ts_utc_ms": ts, "id": coin_id, "from_ms": int(frm_ms), "to_ms": int(to_ms), "data": data}, status

        return self.cache.get_or_fetch(key_parts, fetcher, ttl_s=ttl_s, attempts=self.api_config["coingecko"]["retries"].get("max_attempts", 3))

    @staticmethod
    def to_dataframe(coin_id: str, payload: Dict[str, Any]) -> pd.DataFrame:
        data = payload.get("data", {})
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        vol_map = {int(v[0]): float(v[1]) for v in volumes if isinstance(v, list) and len(v) >= 2}
        rows: List[Dict[str, Any]] = []
        for p in prices:
            try:
                ts_ms = int(p[0])
                price = float(p[1])
                rows.append({
                    "ts_utc_ms": ts_ms,
                    "coin_id": coin_id,
                    "o": price,
                    "h": price,
                    "l": price,
                    "c": price,
                    "volume": vol_map.get(ts_ms),
                    "agg_method": "coingecko_range",
                })
            except Exception:
                continue
        return pd.DataFrame(rows, columns=["ts_utc_ms","coin_id","o","h","l","c","volume","agg_method"])

    def collect(self, coin_id: str, frm_ms: int, to_ms: int, write: bool = True) -> Tuple[pd.DataFrame, int]:
        payload = self.get_range(coin_id, frm_ms, to_ms)
        df = self.to_dataframe(coin_id, payload)
        written = 0
        if write and not df.empty:
            written = self.writer.write("five_min", df, dedup_keys=["coin_id","ts_utc_ms"], partition_cols=None)
        return df, written

    def collect_validation_ranges(self, coin_ids: List[str], frm_ms: int, to_ms: int) -> Tuple[pd.DataFrame, int]:
        """Collecte ranges de validation pour plusieurs coins"""
        all_dfs = []
        total_written = 0
        
        for coin_id in coin_ids:
            try:
                df, written = self.collect(coin_id, frm_ms, to_ms, write=True)
                all_dfs.append(df)
                total_written += written
            except Exception as e:
                print(f"  ❌ Erreur collecte ranges: {e}")
                continue
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            return combined_df, total_written
        return pd.DataFrame(), 0


def main() -> None:
    rc = RangeCollector()
    now = _now_ms()
    frm = now - 24 * 3600 * 1000
    df, written = rc.collect("bitcoin", frm_ms=frm, to_ms=now, write=False)
    print(f"rows={len(df)} written={written}")


if __name__ == "__main__":
    main()
