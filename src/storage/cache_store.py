"""
CacheStore — Cache en mémoire avec TTL, persistance JSON, backoff/jitter et circuit‑breaker.

Objectif
— Fournir aux collecteurs et outils un cache robuste avec:
  - TTL par entrée
  - get_or_fetch(fetcher) avec retry expo + jitter
  - compteurs/historiques légers (hits/misses/latence/erreurs)
  - circuit‑breaker par clé (cooldown après échecs consécutifs)
  - persistance simple dans src/ops/runtime/cache_store.json

Notes
— La valeur est persistée uniquement si JSON‑sérialisable (dict/list/str/num/bool/None).
— Sécurisé par lock (thread‑safe). Un seul process est attendu (spec: single_process_lock).
— Aucune dépendance réseau ici; les collecteurs fournissent le fetcher.
"""

from __future__ import annotations

import json
import os
import time
import math
import hashlib
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional, Tuple

from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type


def _utc_now_s() -> float:
	return time.time()


def _is_json_serializable(obj: Any) -> bool:
	try:
		json.dumps(obj)
		return True
	except Exception:
		return False


def _stable_key(parts: Any) -> str:
	"""Crée une clé stable à partir d'un str/dict/tuple quelconque."""
	if isinstance(parts, str):
		s = parts
	else:
		try:
			s = json.dumps(parts, sort_keys=True, separators=(",", ":"))
		except Exception:
			s = str(parts)
	h = hashlib.sha1(s.encode("utf-8")).hexdigest()
	return h


@dataclass
class CacheEntry:
	value: Any
	expires_at: float
	created_at: float
	meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyStats:
	hits: int = 0
	misses: int = 0
	sets: int = 0
	expirations: int = 0
	failures: int = 0
	last_error: Optional[str] = None
	last_status: Optional[int] = None
	last_latency_ms: Optional[int] = None
	consecutive_failures: int = 0
	circuit_open_until: float = 0.0


class CacheStore:
	def __init__(
		self,
		persist_path: Optional[str] = None,
		default_ttl_s: int = 60,
		circuit_fail_threshold: int = 5,
		circuit_cooldown_s: int = 60,
		max_items: int = 10000,
	) -> None:
		# Réglages
		self.default_ttl_s = int(default_ttl_s)
		self.circuit_fail_threshold = int(circuit_fail_threshold)
		self.circuit_cooldown_s = int(circuit_cooldown_s)
		self.max_items = int(max_items)

		# Dictionnaires en mémoire
		self._store: Dict[str, CacheEntry] = {}
		self._stats: Dict[str, KeyStats] = {}

		# Verrous
		self._lock = threading.RLock()

		# Persistance
		self.persist_path = persist_path or os.path.join(
			os.path.dirname(os.path.dirname(__file__)), "ops", "runtime", "cache_store.json"
		)
		os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
		self._file_lock = FileLock(self.persist_path + ".lock")

		# Chargement initial
		self._load()

	# -------------- Persistance --------------
	def _load(self) -> None:
		with self._lock:
			if not os.path.exists(self.persist_path):
				return
			try:
				with self._file_lock:
					with open(self.persist_path, "r", encoding="utf-8") as f:
						data = json.load(f)
				entries = data.get("entries", {})
				stats = data.get("stats", {})
				now_s = _utc_now_s()
				for k, v in entries.items():
					expires_at = float(v.get("expires_at", 0))
					if expires_at and expires_at <= now_s:
						continue
					self._store[k] = CacheEntry(
						value=v.get("value"),
						expires_at=expires_at,
						created_at=float(v.get("created_at", now_s)),
						meta=v.get("meta", {}),
					)
				# Charger les stats si disponibles
				for k, sv in stats.items():
					try:
						ks = KeyStats(
							hits=int(sv.get("hits", 0)),
							misses=int(sv.get("misses", 0)),
							sets=int(sv.get("sets", 0)),
							expirations=int(sv.get("expirations", 0)),
							failures=int(sv.get("failures", 0)),
							last_error=sv.get("last_error"),
							last_status=sv.get("last_status"),
							last_latency_ms=sv.get("last_latency_ms"),
							consecutive_failures=int(sv.get("consecutive_failures", 0)),
							circuit_open_until=float(sv.get("circuit_open_until", 0.0)),
						)
						self._stats[k] = ks
					except Exception:
						continue
			except Exception:
				# Ignore lecture cassée
				return

	def _save(self) -> None:
		with self._lock:
			now_s = _utc_now_s()
			serializable: Dict[str, Any] = {"version": 2, "saved_at": now_s, "entries": {}, "stats": {}}
			for k, entry in self._store.items():
				if entry.expires_at and entry.expires_at <= now_s:
					continue
				# Ne persist que si JSON‑sérialisable
				if not _is_json_serializable(entry.value):
					continue
				serializable["entries"][k] = {
					"value": entry.value,
					"expires_at": entry.expires_at,
					"created_at": entry.created_at,
					"meta": entry.meta,
				}
			# Sauver un sous‑ensemble utile des stats
			for k, st in self._stats.items():
				serializable["stats"][k] = asdict(st)
			tmp_path = self.persist_path + ".tmp"
			try:
				with self._file_lock:
					with open(tmp_path, "w", encoding="utf-8") as f:
						json.dump(serializable, f, ensure_ascii=False)
					os.replace(tmp_path, self.persist_path)
			finally:
				if os.path.exists(tmp_path):
					try:
						os.remove(tmp_path)
					except Exception:
						pass

	# -------------- Outils --------------
	def _stats_for(self, key: str) -> KeyStats:
		st = self._stats.get(key)
		if st is None:
			st = KeyStats()
			self._stats[key] = st
		return st

	def _is_expired(self, entry: CacheEntry, now_s: Optional[float] = None) -> bool:
		now_s = now_s or _utc_now_s()
		return bool(entry.expires_at and entry.expires_at <= now_s)

	def _evict_if_needed(self) -> None:
		if len(self._store) <= self.max_items:
			return
		# Politique simple: éjecter les plus anciens (created_at)
		items = sorted(self._store.items(), key=lambda kv: kv[1].created_at)
		to_remove = len(self._store) - self.max_items
		for i in range(to_remove):
			self._store.pop(items[i][0], None)

	# -------------- API --------------
	def make_key(self, parts: Any) -> str:
		return _stable_key(parts)

	def get_entry(self, key: str) -> Optional[CacheEntry]:
		with self._lock:
			entry = self._store.get(key)
			st = self._stats_for(key)
			if entry is None:
				st.misses += 1
				return None
			if self._is_expired(entry):
				st.expirations += 1
				self._store.pop(key, None)
				return None
			st.hits += 1
			return entry

	def get(self, key: str) -> Optional[Any]:
		e = self.get_entry(key)
		return e.value if e else None

	def set(self, key: str, value: Any, ttl_s: Optional[int] = None, meta: Optional[Dict[str, Any]] = None) -> None:
		ttl_s = int(ttl_s if ttl_s is not None else self.default_ttl_s)
		now_s = _utc_now_s()
		entry = CacheEntry(value=value, expires_at=(now_s + ttl_s if ttl_s > 0 else 0), created_at=now_s, meta=meta or {})
		with self._lock:
			self._store[key] = entry
			self._stats_for(key).sets += 1
			self._evict_if_needed()
		# Persist en arrière‑plan (ici synchrone pour simplicité et robustesse)
		self._save()

	def delete(self, key: str) -> None:
		with self._lock:
			self._store.pop(key, None)
		self._save()

	def purge_expired(self) -> int:
		removed = 0
		with self._lock:
			now_s = _utc_now_s()
			for k in list(self._store.keys()):
				entry = self._store[k]
				if self._is_expired(entry, now_s):
					self._store.pop(k, None)
					self._stats_for(k).expirations += 1
					removed += 1
		if removed:
			self._save()
		return removed

	def clear(self) -> None:
		with self._lock:
			self._store.clear()
			self._stats.clear()
		self._save()

	def is_circuit_open(self, key: str) -> bool:
		st = self._stats_for(key)
		return st.circuit_open_until > _utc_now_s()

	def _record_failure(self, key: str, err: Exception, status: Optional[int] = None) -> None:
		st = self._stats_for(key)
		st.failures += 1
		st.consecutive_failures += 1
		st.last_error = repr(err)
		st.last_status = status
		if st.consecutive_failures >= self.circuit_fail_threshold:
			st.circuit_open_until = _utc_now_s() + self.circuit_cooldown_s

	def _record_success(self, key: str, latency_ms: Optional[int] = None, status: Optional[int] = None) -> None:
		st = self._stats_for(key)
		st.consecutive_failures = 0
		st.last_error = None
		st.last_status = status
		st.last_latency_ms = latency_ms

	def stats(self, key: Optional[str] = None) -> Dict[str, Any]:
		with self._lock:
			if key is not None:
				st = self._stats.get(key)
				return asdict(st) if st else {}
			return {k: asdict(v) for k, v in self._stats.items()}

	# -------------- get_or_fetch avec retry/backoff --------------
	def get_or_fetch(
		self,
		key_parts: Any,
		fetcher: Callable[[], Tuple[Any, Optional[int]] | Any],
		ttl_s: Optional[int] = None,
		attempts: int = 3,
	) -> Any:
		"""
		Renvoie la valeur cache si valide; sinon appelle fetcher() avec retry expo+jitter.

		fetcher peut renvoyer soit value, soit (value, status_code).
		En cas d'échec après retries, remonte l'exception.
		"""
		key = self.make_key(key_parts)

		# Hit cache
		entry = self.get_entry(key)
		if entry is not None:
			return entry.value

		# Circuit‑breaker
		if self.is_circuit_open(key):
			st = self._stats_for(key)
			raise RuntimeError(f"Circuit ouvert pour la clé; encore {max(0, int(st.circuit_open_until - _utc_now_s()))}s")

		@retry(
			stop=stop_after_attempt(attempts),
			wait=wait_exponential_jitter(exp_base=2, max=30),
			retry=retry_if_exception_type(Exception),
			reraise=True,
		)
		def _run_fetch():
			t0 = time.perf_counter()
			result = fetcher()
			latency_ms = int((time.perf_counter() - t0) * 1000)
			status: Optional[int] = None
			value = result
			if isinstance(result, tuple) and len(result) == 2:
				value, status = result
			# Succès → reset stats, set cache
			self._record_success(key, latency_ms=latency_ms, status=status)
			# Enrichir meta pour inspection hors‑process (status CLI)
			meta = {"key_parts": key_parts, "last_latency_ms": latency_ms, "last_status": status}
			self.set(key, value, ttl_s=ttl_s, meta=meta)
			return value

		try:
			return _run_fetch()
		except Exception as e:
			self._record_failure(key, e)
			raise


__all__ = ["CacheStore", "CacheEntry", "KeyStats"]

