"""
Tickers Collector — CoinGecko /coins/{id}/tickers (TTL 600s)

— Récupère les tickers (USD/USDT) par coin, calcule un spread proxy par exchange
— Whitelist exchanges depuis costs.json (filtered=False si autorisé, True sinon)
— Marque la médiane de spread par coin (median_flag=True pour la ou les lignes au plus proche de la médiane)
— Ecrit dataset=tickers_spread via ParquetWriter avec schéma:
   [ts_utc_ms, coin_id, exch, bid, ask, mid, spread_pct, filtered, median_flag]
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


class TickersCollector:
	TARGETS = {"USD", "USDT"}

	def __init__(self, cache: Optional[CacheStore] = None, writer: Optional[ParquetWriter] = None) -> None:
		self.api_config = _load_json(os.path.join(CONFIG_DIR, "api_config.json"))
		self.coins_config = _load_json(os.path.join(CONFIG_DIR, "coins.json"))
		self.costs = _load_json(os.path.join(CONFIG_DIR, "costs.json"))
		ttl = int(self.api_config["coingecko"]["ttl_sec"].get("tickers", 600))
		self.cache = cache or CacheStore(default_ttl_s=ttl)
		self.writer = writer or ParquetWriter()
		self._rotator = _KeyRotator(self.api_config["coingecko"]["auth"]["env_keys_var"])

		wl = self.costs.get("spread_proxy", {}).get("whitelist_exchanges", [])
		self.whitelist = {w.lower() for w in wl}

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
		return cg["base_url"] + cg["endpoints"]["tickers"].format(id=coin_id)

	def _params(self) -> Dict[str, Any]:
		return {"page": 1}

	def _enabled_coin_ids(self) -> List[str]:
		coins = self.coins_config.get("coins", [])
		return [c["id"] for c in coins if c.get("enabled", True)]

	def fetch_tickers(self, coin_id: str) -> Tuple[Dict[str, Any], int]:
		# Acquire rate limiter slot
		api_acquire("tickers")
		
		cg = self.api_config["coingecko"]
		headers: Dict[str, str] = {}
		api_key = self._rotator.next()
		if api_key:
			headers[cg["auth"]["header_name"]] = api_key
		url = self._endpoint(coin_id)
		params = self._params()
		with self._client() as client:
			r = client.get(url, params=params, headers=headers)
			status = r.status_code
			
			# Register API call result
			api_register("tickers", status)
			
			if status >= 500 or status == 429:
				r.raise_for_status()
			if status >= 400:
				r.raise_for_status()
			data = r.json()
			return data, status

	def get_tickers(self, coin_id: str) -> Dict[str, Any]:
		ttl_s = int(self.api_config["coingecko"]["ttl_sec"].get("tickers", 600))
		key_parts = {"endpoint": "tickers", "id": coin_id}

		def fetcher():
			data, status = self.fetch_tickers(coin_id)
			ts = _now_ms()
			return {"ts_utc_ms": ts, "id": coin_id, "data": data}, status

		return self.cache.get_or_fetch(key_parts, fetcher, ttl_s=ttl_s, attempts=self.api_config["coingecko"]["retries"].get("max_attempts", 3))

	@staticmethod
	def _as_fraction(pct: Optional[float]) -> Optional[float]:
		if pct is None:
			return None
		try:
			v = float(pct)
		except Exception:
			return None
		# Si fourni comme pourcentage (ex 0.5 pour 0.5%), ramener en fraction
		return v / 100.0 if v > 1 else v

	def to_dataframe(self, coin_id: str, payload: Dict[str, Any]) -> pd.DataFrame:
		ts = int(payload.get("ts_utc_ms", _now_ms()))
		data = payload.get("data", {})
		tickers = data.get("tickers", [])
		rows: List[Dict[str, Any]] = []
		for t in tickers:
			try:
				target = t.get("target")
				if target not in self.TARGETS:
					continue
				mkt = t.get("market", {})
				exch = (mkt.get("name") or "").strip()
				if not exch:
					continue
				spread_pct = self._as_fraction(t.get("bid_ask_spread_percentage"))
				# prix de référence (mid): converted_last.usd si dispo, sinon last si target=USD
				mid = None
				converted_last = t.get("converted_last") or {}
				if isinstance(converted_last, dict) and isinstance(converted_last.get("usd"), (int, float)):
					mid = float(converted_last["usd"])
				elif target == "USD" and isinstance(t.get("last"), (int, float)):
					mid = float(t.get("last"))

				bid = ask = None
				if mid is not None and isinstance(spread_pct, (int, float)):
					half = float(spread_pct) / 2.0
					bid = mid * (1 - half)
					ask = mid * (1 + half)

				filtered = exch.lower() not in self.whitelist if self.whitelist else False

				rows.append({
					"ts_utc_ms": ts,
					"coin_id": coin_id,
					"exch": exch,
					"bid": bid,
					"ask": ask,
					"mid": mid,
					"spread_pct": float(spread_pct) if spread_pct is not None else None,
					"filtered": bool(filtered),
					"median_flag": False,  # rempli après calcul médiane
				})
			except Exception:
				continue

		df = pd.DataFrame(rows, columns=["ts_utc_ms","coin_id","exch","bid","ask","mid","spread_pct","filtered","median_flag"])
		if df.empty:
			return df
		# Calcul médiane du spread sur les non filtrés
		try:
			cand = df[(df["filtered"] == False) & df["spread_pct"].notna()]  # noqa: E712
			if not cand.empty:
				med = float(cand["spread_pct"].median())
				# flag sur la/les lignes les plus proches de la médiane
				df.loc[cand.index, "median_flag"] = (cand["spread_pct"] - med).abs() == (cand["spread_pct"] - med).abs().min()
		except Exception:
			pass
		return df

	def collect(self, coin_ids: Optional[List[str]] = None, write: bool = True) -> Tuple[pd.DataFrame, int]:
		ids = coin_ids or self._enabled_coin_ids()
		frames: List[pd.DataFrame] = []
		for cid in ids:
			payload = self.get_tickers(cid)
			df = self.to_dataframe(cid, payload)
			if not df.empty:
				frames.append(df)
		written = 0
		if frames and write:
			out = pd.concat(frames, ignore_index=True)
			written = self.writer.write("tickers_spread", out, dedup_keys=["coin_id","exch","ts_utc_ms"], partition_cols=None)
			return out, written
		return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts_utc_ms","coin_id","exch","bid","ask","mid","spread_pct","filtered","median_flag"]), written)


def main() -> None:
	tc = TickersCollector()
	df, written = tc.collect()
	print(f"rows={len(df)} written={written}")


if __name__ == "__main__":
	main()
