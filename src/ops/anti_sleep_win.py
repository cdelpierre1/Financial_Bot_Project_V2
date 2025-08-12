"""
Anti-sommeil Windows (MVP)

Empêche la mise en veille pendant l'exécution du bot en utilisant
SetThreadExecutionState via ctypes. No-op sur plateformes non Windows
ou en cas d'échec.
"""
from __future__ import annotations

import os
from typing import Optional

try:
	import ctypes  # type: ignore
except Exception:  # pragma: no cover
	ctypes = None  # type: ignore


# Flags Windows
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040
ES_CONTINUOUS = 0x80000000


def _is_windows() -> bool:
	return os.name == "nt"


def _set_exec_state(flags: int) -> bool:
	if not _is_windows() or ctypes is None:
		return False
	try:
		res = ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]
		return bool(res)
	except Exception:
		return False


class AntiSleepGuard:
	"""Garde anti-sommeil simple pour Windows.

	start(): active ES_CONTINUOUS | ES_SYSTEM_REQUIRED (+ option AWAYMODE)
	stop(): rétablit ES_CONTINUOUS
	"""

	def __init__(self, enabled: bool = True, use_away_mode: bool = False) -> None:
		self.enabled = bool(enabled)
		self.use_away_mode = bool(use_away_mode)
		self._active = False

	def start(self) -> bool:
		if not self.enabled or self._active:
			return False
		flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
		if self.use_away_mode:
			flags |= ES_AWAYMODE_REQUIRED
		ok = _set_exec_state(flags)
		self._active = True
		return ok

	def stop(self) -> bool:
		if not self._active:
			return False
		ok = _set_exec_state(ES_CONTINUOUS)
		self._active = False
		return ok

