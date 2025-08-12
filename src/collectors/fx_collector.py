"""
FX Collector — Taux USD par EUR (TTL 3600s)

— Par défaut utilise ExchangeRate-API (latest/EUR) avec clé FX_API_KEY.
— Fallback possible: exchangerate.host (sans clé) si configuré.
— Cache via CacheStore; retries expo+jitter 5xx; 4xx non retentés.
— Ecrit dataset=fx via ParquetWriter avec schéma: [ts_utc_ms, base, rate_usd_per_eur, ttl_until].
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple
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


def _load_fx_key(env_var: str) -> Optional[str]:
	k = os.environ.get(env_var)
	if k:
		return k
	env_path = os.path.join(CONFIG_DIR, "secrets", ".env.local")
	if os.path.exists(env_path):
		vals = dotenv_values(env_path)
		v = vals.get(env_var)
		if v:
			return v
	return None


class FxCollector:
	def __init__(self, cache: Optional[CacheStore] = None, writer: Optional[ParquetWriter] = None) -> None:
		self.api_config = _load_json(os.path.join(CONFIG_DIR, "api_config.json"))
		ttl = int(self.api_config["fx"].get("ttl_sec", 3600))
		self.cache = cache or CacheStore(default_ttl_s=ttl)
		self.writer = writer or ParquetWriter()

	def _client(self) -> httpx.Client:
		timeouts = self.api_config["fx"].get("timeouts_sec", {"connect": 5, "read": 10})
		connect = float(timeouts.get("connect", 5))
		read = float(timeouts.get("read", 10))
		write = float(timeouts.get("write", read))
		pool = float(timeouts.get("pool", read))
		return httpx.Client(timeout=httpx.Timeout(connect=connect, read=read, write=write, pool=pool))

	def _provider(self) -> str:
		return self.api_config["fx"].get("provider", "exchangerate-api")

	def _endpoint_and_headers(self) -> Tuple[str, Dict[str, str]]:
		prov = self._provider()
		if prov == "exchangerate-api":
			base = self.api_config["fx"]["exchangerate_api"]["base_url"]
			endpoint = self.api_config["fx"]["exchangerate_api"]["endpoint_latest"]
			env_var = self.api_config["fx"]["exchangerate_api"]["env_api_key_var"]
			api_key = _load_fx_key(env_var)
			if not api_key:
				raise RuntimeError("FX_API_KEY manquante pour ExchangeRate-API")
			url = base + endpoint.replace("{api_key}", api_key)
			return url, {}
		else:
			base = self.api_config["fx"]["exchangerate_host"]["base_url"]
			endpoint = self.api_config["fx"]["exchangerate_host"]["endpoint_latest"]
			url = base + endpoint
			return url, {}

	def fetch_latest(self) -> Tuple[Dict[str, Any], int]:
		# Acquire rate limiter slot
		api_acquire("fx_latest")
		
		url, headers = self._endpoint_and_headers()
		with self._client() as client:
			r = client.get(url, headers=headers)
			status = r.status_code
			
			# Register API call result
			api_register("fx_latest", status)
			
			# Retry seulement pour 5xx; 4xx → direct
			if status >= 500:
				r.raise_for_status()
			if status >= 400:
				r.raise_for_status()
			data = r.json()
			return data, status

	def get_latest(self) -> Dict[str, Any]:
		ttl_s = int(self.api_config["fx"].get("ttl_sec", 3600))
		key_parts = {"endpoint": "fx_latest", "base": "EUR"}

		def fetcher():
			data, status = self.fetch_latest()
			ts = _now_ms()
			return {"ts_utc_ms": ts, "data": data}, status

		return self.cache.get_or_fetch(key_parts, fetcher, ttl_s=ttl_s, attempts=self.api_config["fx"]["retries"].get("max_attempts", 3))

	@staticmethod
	def to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
		ts = int(payload.get("ts_utc_ms", _now_ms()))
		data = payload.get("data", {})
		base = "EUR"

		# ExchangeRate-API: {conversion_rates:{"USD": 1.09, ...}}
		rate_usd: Optional[float] = None
		if "conversion_rates" in data:
			try:
				v = data["conversion_rates"].get("USD")
				if isinstance(v, (int, float)):
					rate_usd = float(v)
			except Exception:
				rate_usd = None
		# exchangerate.host: {rates:{"USD":1.09}, base:"EUR"}
		if rate_usd is None and "rates" in data:
			try:
				v = data["rates"].get("USD")
				if isinstance(v, (int, float)):
					rate_usd = float(v)
			except Exception:
				rate_usd = None

		ttl_until = ts + int(payload.get("ttl_s", 0)) * 1000 if "ttl_s" in payload else ts + 3600 * 1000

		row = {
			"ts_utc_ms": ts,
			"base": base,
			"rate_usd_per_eur": rate_usd,
			"ttl_until": ttl_until,
		}
		return pd.DataFrame([row], columns=["ts_utc_ms","base","rate_usd_per_eur","ttl_until"])

	def collect(self, write: bool = True) -> Tuple[pd.DataFrame, int]:
		payload = self.get_latest()
		df = self.to_dataframe(payload)
		written = 0
		if write and not df.empty:
			written = self.writer.write("fx", df, dedup_keys=["base","ts_utc_ms"], partition_cols=None)
		return df, written


def main() -> None:
	fxc = FxCollector()
	df, written = fxc.collect()
	print(f"rows={len(df)} written={written}")


if __name__ == "__main__":
	main()

