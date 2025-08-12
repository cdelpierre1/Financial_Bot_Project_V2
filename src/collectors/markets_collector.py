"""
Markets Collector — CoinGecko coins/markets (TTL 60s)

– Récupère prix USD, market cap, volume 24h, variations 1h/24h/7j
– Cache TTL via CacheStore; retries expo+jitter sur 429/5xx; rotation de clés
– Ecrit en Parquet (dataset=markets) avec ParquetWriter

Schéma de sortie (DataFrame):
columns = ["ts_utc_ms","coin_id","price_usd","market_cap","vol24h","chg1h","chg24h","chg7d"]
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
import pandas as pd
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


class MarketsCollector:
	def __init__(self, cache: Optional[CacheStore] = None, writer: Optional[ParquetWriter] = None) -> None:
		self.api_config = _load_json(os.path.join(CONFIG_DIR, "api_config.json"))
		self.coins_config = _load_json(os.path.join(CONFIG_DIR, "coins.json"))
		self.cache = cache or CacheStore(default_ttl_s=int(self.api_config["coingecko"]["ttl_sec"].get("coins_markets", 60)))
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

	def _endpoint(self) -> str:
		cg = self.api_config["coingecko"]
		return cg["base_url"] + cg["endpoints"]["coins_markets"]

	def _default_params(self) -> Dict[str, Any]:
		cg = self.api_config["coingecko"]
		return {
			"vs_currency": cg["default_params"].get("vs_currency", "usd"),
			"order": cg["default_params"].get("order", "market_cap_desc"),
			"per_page": cg["default_params"].get("per_page", 100),
			"page": cg["default_params"].get("page", 1),
			"price_change_percentage": cg["default_params"].get("price_change_percentage", "1h,24h,7d"),
		}

	def _enabled_coin_ids(self) -> List[str]:
		coins = self.coins_config.get("coins", [])
		return [c["id"] for c in coins if c.get("enabled", True)]

	def fetch_markets(self, coin_ids: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], int]:
		ids = coin_ids or self._enabled_coin_ids()
		ids = sorted(set(ids))
		if not ids:
			return [], 200

		# Acquire rate limiter slot
		api_acquire("coins_markets")

		cg = self.api_config["coingecko"]
		headers: Dict[str, str] = {}
		api_key = self._rotator.next()
		if api_key:
			headers[cg["auth"]["header_name"]] = api_key
		params = self._default_params() | {"ids": ",".join(ids)}
		url = self._endpoint()

		with self._client() as client:
			r = client.get(url, params=params, headers=headers)
			status = r.status_code
			
			# Register API call result
			api_register("coins_markets", status)
			
			if status >= 500 or status == 429:
				r.raise_for_status()
			if status >= 400:
				r.raise_for_status()
			data = r.json()
			# données: liste de marchés par coin
			return data, status

	def get_markets(self, coin_ids: Optional[List[str]] = None) -> Dict[str, Any]:
		ids = coin_ids or self._enabled_coin_ids()
		ids = sorted(set(ids))
		ttl_s = int(self.api_config["coingecko"]["ttl_sec"].get("coins_markets", 60))
		key_parts = {"endpoint": "coins_markets", "ids": ids, "vs": "usd"}

		def fetcher():
			data, status = self.fetch_markets(ids)
			ts = _now_ms()
			return {"ts_utc_ms": ts, "data": data}, status

		return self.cache.get_or_fetch(key_parts, fetcher, ttl_s=ttl_s, attempts=self.api_config["coingecko"]["retries"].get("max_attempts", 3))

	@staticmethod
	def to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
		ts = int(payload.get("ts_utc_ms", _now_ms()))
		data = payload.get("data", [])
		rows: List[Dict[str, Any]] = []
		for item in data:
			try:
				coin_id = item.get("id")
				price = item.get("current_price")
				market_cap = item.get("market_cap")
				vol = item.get("total_volume")
				ch1 = item.get("price_change_percentage_1h_in_currency")
				ch24 = item.get("price_change_percentage_24h_in_currency")
				ch7 = item.get("price_change_percentage_7d_in_currency")
				if coin_id is None:
					continue
				rows.append({
					"ts_utc_ms": ts,
					"coin_id": coin_id,
					"price_usd": float(price) if isinstance(price, (int, float)) else None,
					"market_cap": float(market_cap) if isinstance(market_cap, (int, float)) else None,
					"vol24h": float(vol) if isinstance(vol, (int, float)) else None,
					"chg1h": float(ch1) if isinstance(ch1, (int, float)) else None,
					"chg24h": float(ch24) if isinstance(ch24, (int, float)) else None,
					"chg7d": float(ch7) if isinstance(ch7, (int, float)) else None,
				})
			except Exception:
				continue
		return pd.DataFrame(rows, columns=["ts_utc_ms","coin_id","price_usd","market_cap","vol24h","chg1h","chg24h","chg7d"])

	def collect(self, coin_ids: Optional[List[str]] = None, write: bool = True) -> Tuple[pd.DataFrame, int]:
		payload = self.get_markets(coin_ids)
		df = self.to_dataframe(payload)
		written = 0
		if write and not df.empty:
			# Dédup par (coin_id, ts_utc_ms)
			written = self.writer.write("markets", df, dedup_keys=["coin_id","ts_utc_ms"], partition_cols=None)
		return df, written


def main() -> None:
	mc = MarketsCollector()
	df, written = mc.collect()
	print(f"rows={len(df)} written={written}")


if __name__ == "__main__":
	main()

