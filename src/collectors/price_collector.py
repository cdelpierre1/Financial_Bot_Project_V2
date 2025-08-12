"""
Price Collector — CoinGecko simple/price (TTL 10s)

Fonctions principales:
- get_prices(coin_ids): récupère le prix USD courant des coins via /simple/price
- Rotation round‑robin des clés CoinGecko (x-cg-pro-api-key)
- TTL 10s via CacheStore pour éviter la sur‑sollicitation

Sortie:
{
  "ts_utc_ms": <int>,
  "prices": { coin_id: price_usd, ... }
}
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import httpx
import time
from dotenv import dotenv_values

from storage.cache_store import CacheStore
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
		# 1) Essaye l'env
		keys_env = os.environ.get(env_var)
		keys: List[str] = []
		if keys_env:
			keys = [k.strip() for k in keys_env.split(",") if k.strip()]
		# 2) Essaye .env.local si vide
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


class PriceCollector:
	def __init__(self, cache: Optional[CacheStore] = None) -> None:
		self.api_config = _load_json(os.path.join(CONFIG_DIR, "api_config.json"))
		self.coins_config = _load_json(os.path.join(CONFIG_DIR, "coins.json"))
		self.cache = cache or CacheStore(default_ttl_s=10)
		self._rotator = _KeyRotator(self.api_config["coingecko"]["auth"]["env_keys_var"])

	def _client(self) -> httpx.Client:
		timeouts = self.api_config["coingecko"].get("timeouts_sec", {"connect": 5, "read": 10})
		# httpx requires either a default or all four timeout parameters
		connect = float(timeouts.get("connect", 5))
		read = float(timeouts.get("read", 10))
		write = float(timeouts.get("write", read))
		pool = float(timeouts.get("pool", read))
		headers = {"User-Agent": self.api_config["coingecko"].get("user_agent", "fb2/1.0")}
		return httpx.Client(timeout=httpx.Timeout(connect=connect, read=read, write=write, pool=pool), headers=headers)

	def _endpoint(self) -> str:
		cg = self.api_config["coingecko"]
		return cg["base_url"] + cg["endpoints"]["simple_price"]

	def _default_params(self) -> Dict[str, Any]:
		return {
			"vs_currencies": self.api_config["coingecko"]["default_params"].get("vs_currency", "usd"),
			"precision": self.api_config["coingecko"]["default_params"].get("precision", "full"),
		}

	def _enabled_coin_ids(self) -> List[str]:
		coins = self.coins_config.get("coins", [])
		return [c["id"] for c in coins if c.get("enabled", True)]

	def get_prices(self, coin_ids: Optional[List[str]] = None) -> Dict[str, Any]:
		ids = coin_ids or self._enabled_coin_ids()
		ids = sorted(set(ids))
		if not ids:
			return {"ts_utc_ms": _now_ms(), "prices": {}}

		cg = self.api_config["coingecko"]
		ttl_s = int(cg["ttl_sec"].get("simple_price", 10))

		key_parts = {"endpoint": "simple_price", "ids": ids, "vs": "usd"}

		def fetcher():
			headers = {}
			api_key = self._rotator.next()
			if api_key:
				headers[cg["auth"]["header_name"]] = api_key
			params = self._default_params() | {"ids": ",".join(ids)}
			url = self._endpoint()
			# Rate limiter global
			if not api_acquire("simple_price"):
				raise RuntimeError("RATE_LIMIT_LOCAL: simple_price minute bucket plein")
			with self._client() as client:
				r = client.get(url, params=params, headers=headers)
				status = r.status_code
				api_register("simple_price", status)
				if status == 429:
					# Ne pas bloquer 60s: remonter pour replanification externe
					r.raise_for_status()
				if status >= 500:
					r.raise_for_status()
				if status >= 400:
					r.raise_for_status()
				data = r.json()
				prices: Dict[str, float] = {}
				for cid in ids:
					v = data.get(cid)
					if isinstance(v, dict):
						price = v.get("usd")
						if isinstance(price, (int, float)):
							prices[cid] = float(price)
				return {"ts_utc_ms": _now_ms(), "prices": prices}, status

		result = self.cache.get_or_fetch(key_parts, fetcher, ttl_s=ttl_s, attempts=self.api_config["coingecko"]["retries"].get("max_attempts", 3))
		return result


def main() -> None:
	pc = PriceCollector()
	out = pc.get_prices()
	print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
	main()

