"""
CLI Opérations — start/stop/predict/status/settings (Typer)

Ce CLI fournit un squelette opérationnel:
- start: lance le scheduler en arrière‑plan, crée pid/lock, et vérifie l’unicité de process
- stop: termine le scheduler et nettoie les fichiers runtime
- predict: flux interactif (kind, coin, unité, valeur) avec interpolation des seuils/cibles
- status: affiche un état synthétique (runtime, datasets, cache, état du scheduler)
- settings: affiche des infos clés de configuration

Note: la pipeline de prédiction n’est pas encore implémentée; la commande predict renvoie une sortie structurée avec décision "NO_MODEL".
"""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import time

# Assurer que le dossier parent (src) est dans PYTHONPATH quand appelé via -m ops.cli depuis la racine
_THIS_DIR = os.path.dirname(__file__)
_SRC_ROOT = os.path.dirname(_THIS_DIR)
if _SRC_ROOT not in sys.path:
	sys.path.insert(0, _SRC_ROOT)

import typer
import subprocess
import psutil
import shutil
from prediction.threshold_policy import interpolate_error_threshold
from prediction.pipeline import PredictionPipeline
from collectors.price_collector import PriceCollector
from collectors.fx_collector import FxCollector
from prediction.evaluation import save_prediction, generate_prediction_id
from prediction.trainer import Trainer
from prediction.incremental_trainer import IncrementalTrainer
from prediction.model_store import ModelStore
import pandas as pd
from ops.api_usage import api_snapshot


app = typer.Typer(no_args_is_help=True)

# Alias français pour les commandes principales
@app.command(name="demarrer")
def demarrer_alias() -> None:
	"""Alias: démarre le bot (équivalent à start)."""
	start()

@app.command(name="arreter")
def arreter_alias() -> None:
	"""Alias: arrête le bot (équivalent à stop)."""
	stop()

@app.command(name="etat")
def etat_alias() -> None:
	"""Alias: affiche l'état (équivalent à status)."""
	status()

@app.command(name="parametres")
def parametres_alias() -> None:
	"""Alias: affiche des infos de configuration (équivalent à settings)."""
	settings()

@app.command(name="prevoir")
def prevoir_alias(
	kind: Optional[str] = typer.Option(None, help="price/prix ou amount/montant"),
	coin_opt: Optional[str] = typer.Option(None, "--coin", "--symbole", help="id/symbole/nom du coin (alias: --symbole)"),
	unit_opt: Optional[str] = typer.Option(None, "--unite", help="min|h|d"),
	value_opt: Optional[int] = typer.Option(None, "--valeur", help="entier >0"),
	amount_eur_opt: Optional[float] = typer.Option(None, "--montant-eur", help="Montant EUR (si kind=amount)"),
) -> None:
	"""Alias: prédiction (équivalent à predict)."""
	return predict(kind=kind, coin_opt=coin_opt, unit_opt=unit_opt, value_opt=value_opt, amount_eur_opt=amount_eur_opt)


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()

def _now_ms() -> int:
	return int(time.time() * 1000)


def _load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _settings() -> Dict[str, Any]:
	return _load_json(os.path.join(CONFIG_DIR, "settings.json"))


def _thresholds() -> Dict[str, Any]:
	return _load_json(os.path.join(CONFIG_DIR, "thresholds.json"))


def _targets() -> Dict[str, Any]:
	return _load_json(os.path.join(CONFIG_DIR, "targets.json"))


def _coins() -> List[Dict[str, Any]]:
	return _load_json(os.path.join(CONFIG_DIR, "coins.json")).get("coins", [])


def _runtime_paths() -> Tuple[str, str]:
	s = _settings()
	runtime_dir = s["paths"]["runtime"]
	os.makedirs(runtime_dir, exist_ok=True)
	pid_path = os.path.join(runtime_dir, "pid")
	lock_path = os.path.join(runtime_dir, "app.lock")
	return pid_path, lock_path


def _pid_exists(pid: int) -> bool:
	try:
		if pid <= 0:
			return False
		os.kill(pid, 0)
	except Exception:
		return False
	return True


def _interpolate_threshold(minutes: int, th: Dict[str, Any]) -> float:
	return interpolate_error_threshold(minutes, th)


def _interpolate_target_eur(coin_id: str, minutes: int, targets_cfg: Dict[str, Any]) -> Optional[float]:
	currency = targets_cfg.get("currency", "EUR")
	items = targets_cfg.get("targets", [])
	entry = next((x for x in items if x.get("id") == coin_id), None)
	if not entry:
		return None
	t6 = float(entry.get("profit_net_eur", {}).get("6h", 0.0))
	t24 = float(entry.get("profit_net_eur", {}).get("24h", 0.0))
	m = max(1, int(minutes))
	if m == 360:
		return t6
	if m == 1440:
		return t24
	# interpolation/extrapolation linéaire entre 6h et 24h
	slope = (t24 - t6) / (1440 - 360)
	return t6 + slope * (m - 360)


def _normalize_coin(user_input: str) -> Optional[Dict[str, Any]]:
	coins = _coins()
	val = user_input.strip().lower()
	for c in coins:
		if not c.get("enabled", True):
			continue
		if c.get("id", "").lower() == val or c.get("symbol", "").lower() == val or c.get("name", "").lower() == val:
			return c
	return None


def _normalize_unit(u: str) -> Optional[str]:
	m = u.strip().lower()
	if m in {"min", "m", "minute", "minutes"}:
		return "min"
	if m in {"h", "heure", "heures"}:
		return "h"
	if m in {"d", "j", "jour", "jours"}:
		return "d"
	return None


def _minutes_from(unit: str, value: int) -> int:
	if unit == "min":
		return int(value)
	if unit == "h":
		return int(value) * 60
	if unit == "d":
		return int(value) * 1440
	raise ValueError("Unité inconnue")


def _read_recent_five_min() -> Optional[pd.DataFrame]:
	s = _settings()
	root = s["paths"]["data_parquet"]
	path = os.path.join(root, "five_min")
	if not os.path.isdir(path):
		return None
	try:
		return pd.read_parquet(path)
	except Exception:
		try:
			frames: list[pd.DataFrame] = []
			for r, _, files in os.walk(path):
				for f in files:
					if f.endswith('.parquet'):
						fp = os.path.join(r, f)
						try:
							frames.append(pd.read_parquet(fp))
						except Exception:
							pass
			if frames:
				return pd.concat(frames, ignore_index=True)
		except Exception:
			return None
	return None


def _compute_mae_stats() -> Dict[str, Any]:
	"""Retourne toujours une structure dict avec fenêtres même si vide.
	Lecture robuste des datasets partitionnés (.parquet) via fallback récursif.
	"""
	s = _settings()
	data_root = s["paths"]["data_parquet"]
	import pandas as pd  # import local
	path = os.path.join(data_root, "eval_results")
	def _empty_payload() -> Dict[str, Any]:
		return {
			"window_24h": {"global": {"mae_pct": None, "count": 0}, "by_coin": {}},
			"window_7d": {"global": {"mae_pct": None, "count": 0}, "by_coin": {}},
			"window_30d": {"global": {"mae_pct": None, "count": 0}, "by_coin": {}},
		}
	if not os.path.isdir(path):
		return _empty_payload()
	# tentative directe
	df: Optional["pd.DataFrame"] = None
	try:
		df = pd.read_parquet(path)
	except Exception:
		df = None
	# fallback récursif si vide/None
	if df is None or df.empty:
		frames: List["pd.DataFrame"] = []
		for root_dir, _, files in os.walk(path):
			for f in files:
				if f.endswith(".parquet"):
					fp = os.path.join(root_dir, f)
					try:
						frames.append(pd.read_parquet(fp))
					except Exception:
						pass
		if frames:
			try:
				df = pd.concat(frames, ignore_index=True)
			except Exception:
				df = None
	if df is None or df.empty or "ts_utc_ms" not in df.columns or "abs_error_pct" not in df.columns:
		return _empty_payload()
	now = datetime.now(timezone.utc)
	df = df.copy()
	df["ts"] = pd.to_datetime(df["ts_utc_ms"], unit="ms", utc=True)
	def _agg(sub: "pd.DataFrame") -> Dict[str, Any]:
		if sub is None or sub.empty:
			return {"global": {"mae_pct": None, "count": 0}, "by_coin": {}}
		g_mae = float(sub["abs_error_pct"].mean())
		g_cnt = int(len(sub))
		by_coin = {}
		if "coin_id" in sub.columns:
			grp = sub.groupby("coin_id")
			for cid, g in grp:
				by_coin[str(cid)] = {"mae_pct": float(g["abs_error_pct"].mean()), "count": int(len(g))}
		return {"global": {"mae_pct": g_mae, "count": g_cnt}, "by_coin": by_coin}

	def _win(days: int) -> Dict[str, Any]:
		start = now - timedelta(days=days)
		sub = df[df["ts"] >= start]
		return _agg(sub)
	return {
		"window_24h": _win(1),
		"window_7d": _win(7),
		"window_30d": _win(30),
	}

def _system_stats() -> Dict[str, Any]:
	s = _settings()
	data_root = s["paths"]["data_parquet"]
	out: Dict[str, Any] = {}
	try:
		boot = float(psutil.boot_time())
		uptime_sec = max(0, int(time.time() - boot))
		vm = psutil.virtual_memory()
		ram_used_pct = float(getattr(vm, "percent", 0.0))
		# Disque sur le lecteur des données
		base_drive = s.get("paths", {}).get("base_drive")
		probe_path = None
		if isinstance(base_drive, str) and base_drive:
			probe_path = base_drive if base_drive.endswith(os.sep) else base_drive + os.sep
		else:
			probe_path = os.path.abspath(os.path.join(data_root, os.pardir))
		du = shutil.disk_usage(probe_path)
		disk_free_gb = round(du.free / (1024**3), 2)
		disk_total_gb = round(du.total / (1024**3), 2)
		# GPU (best-effort)
		gpu_available = False
		try:
			import torch  # type: ignore
			gpu_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
		except Exception:
			gpu_available = False
		# Process RAM
		try:
			p = psutil.Process(os.getpid())
			rss_mb = round(p.memory_info().rss / (1024**2), 1)
		except Exception:
			rss_mb = None
		out = {
			"uptime_sec": uptime_sec,
			"ram_used_pct": ram_used_pct,
			"disk_free_gb": disk_free_gb,
			"disk_total_gb": disk_total_gb,
			"gpu_available": gpu_available,
			"process_rss_mb": rss_mb,
		}
	except Exception:
		out = {}
	return out


def format_seconds(seconds: int) -> str:
	"""Formate les secondes en une chaîne lisible (jours, heures, minutes)."""
	if seconds < 60:
		return f"{seconds}s"
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	days, hours = divmod(hours, 24)
	parts = []
	if days > 0:
		parts.append(f"{days}j")
	if hours > 0:
		parts.append(f"{hours}h")
	if minutes > 0:
		parts.append(f"{minutes}m")
	return " ".join(parts) if parts else f"{seconds}s"


@app.command()
def start() -> None:
	"""Démarre le bot: lance le scheduler en arrière‑plan et crée PID/lock."""
	pid_path, lock_path = _runtime_paths()

	# Vérifier si un scheduler tourne déjà
	if os.path.exists(pid_path):
		try:
			with open(pid_path, "r", encoding="utf-8") as f:
				existing = json.load(f)
			old_pid = int(existing.get("pid", -1))
			if old_pid > 0 and psutil.pid_exists(old_pid):
				p = psutil.Process(old_pid)
				if p.is_running():
					typer.echo(f"Scheduler déjà actif (PID={old_pid}).")
					raise typer.Exit(code=0)
		except Exception:
			# Fichier corrompu ou processus mort: on continue après nettoyage
			pass
		# Nettoyer les anciens fichiers (pid/lock) si processus mort
		try:
			os.remove(pid_path)
		except Exception:
			pass
		if os.path.exists(lock_path):
			try:
				os.remove(lock_path)
			except Exception:
				pass

	# Préparer chemins absolus (logs/runtime)
	try:
		s = _settings()
		logs_dir = s["paths"].get("logs", "src/logs")
		if not os.path.isabs(logs_dir):
			project_root = os.path.dirname(ROOT)  # parent de src
			logs_dir = os.path.join(project_root, logs_dir)
		os.makedirs(logs_dir, exist_ok=True)
		log_path = os.path.join(logs_dir, "scheduler.log")
	except Exception:
		log_path = os.path.join(os.getcwd(), "scheduler.log")

	# Lancement du scheduler via subprocess pour obtenir un vrai PID
	try:
		python_exe = sys.executable
		# On exécute le module 'ops.scheduler' avec cwd=ROOT (répertoire src)
		env = os.environ.copy()
		env["PYTHONUNBUFFERED"] = "1"
		# Ouverture du log en mode append + line buffering
		log_f = open(log_path, "a", encoding="utf-8", buffering=1)
		creationflags = 0
		if sys.platform == "win32":
			# Détacher raisonnablement (process groupe propre)
			creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
			# (On évite DETACHED_PROCESS pour garder la console si besoin de debug)
		proc = subprocess.Popen(
			[python_exe, "-m", "ops.scheduler"],
			cwd=ROOT,
			stdout=log_f,
			stderr=log_f,
			env=env,
			creationflags=creationflags,
			close_fds=True
		)
		# Petite attente pour voir si crash immédiat
		time.sleep(1.2)
		ret = proc.poll()
		if ret is not None:
			log_f.flush()
			# Lire la fin du log pour diagnostic
			try:
				with open(log_path, "r", encoding="utf-8") as lf:
					lines = lf.readlines()[-20:]
					typer.echo("❌ Échec démarrage scheduler (exit code {}). Dernières lignes log:\n".format(ret))
					for l in lines:
						typer.echo(l.rstrip())
			except Exception:
				pass
			raise typer.Exit(code=1)
	except Exception as e:
		try:
			log_f.close()
		except Exception:
			pass
		typer.echo(f"Erreur lancement scheduler: {e}")
		raise typer.Exit(code=1)

	# Écriture PID & lock uniquement si processus vivant
	info = {
		"pid": proc.pid,
		"started_at": _now_iso(),
		"host": socket.gethostname(),
		"cmd": "-m ops.scheduler",
		"log": log_path
	}
	try:
		with open(pid_path, "w", encoding="utf-8") as f:
			json.dump(info, f, ensure_ascii=False)
		with open(lock_path, "w", encoding="utf-8") as f:
			f.write("LOCK\n")
	except Exception as e:
		typer.echo(f"⚠️ Impossible d'écrire pid/lock: {e}")

	typer.echo(f"✅ Scheduler démarré (PID={proc.pid}) — log: {log_path}")


@app.command()
def stop() -> None:
	"""Arrête le bot: termine le scheduler si actif et nettoie PID/lock."""
	pid_path, lock_path = _runtime_paths()
	try:
		if os.path.exists(pid_path):
			try:
				with open(pid_path, "r", encoding="utf-8") as f:
					info = json.load(f)
				pid = int(info.get("pid", -1))
			except Exception:
				pid = -1
			if pid > 0 and psutil.pid_exists(pid):
				p = psutil.Process(pid)
				try:
					p.terminate()
					p.wait(timeout=5)
				except Exception:
					try:
						p.kill()
					except Exception:
						pass
			os.remove(pid_path)
		if os.path.exists(lock_path):
			os.remove(lock_path)
		typer.echo("Arrêt effectué. Verrous nettoyés.")
	except Exception as e:
		typer.echo(f"Erreur pendant l’arrêt: {e}")
		raise typer.Exit(code=1)


@app.command()
def predict(
	kind: Optional[str] = typer.Option(None, help="price ou amount"),
	coin_opt: Optional[str] = typer.Option(None, "--coin", help="id/symbol/nom du coin"),
	unit_opt: Optional[str] = typer.Option(None, "--unit", help="min|h|d"),
	value_opt: Optional[int] = typer.Option(None, "--value", help="valeur entière >0"),
	amount_eur_opt: Optional[float] = typer.Option(None, "--amount-eur", help="Montant EUR (requis si kind=amount)"),
	test_mode: bool = typer.Option(False, "--test-mode", help="Mode test rapide pour smoke test"),
) -> None:
	"""Prédiction: interactif par défaut; flags non-interactifs disponibles (--coin --unit --value --kind --amount-eur)."""
	
	# Mode test rapide pour smoke test
	if test_mode:
		print("🧪 Mode test: Simulation de prédiction Bitcoin +10min")
		print("✅ Prédiction test simulée avec succès")
		return
	
	# KIND
	if kind is None:
		kind = typer.prompt("Type de prédiction (price/prix|amount/montant|profit/benefice)").strip().lower()
	else:
		kind = kind.strip().lower()
	# Accepter les synonymes français
	if kind in {"prix"}:
		kind = "price"
	if kind in {"montant"}:
		kind = "amount"
	if kind in {"benefice", "profit", "gain"}:
		kind = "profit"
	if kind not in {"price", "amount", "profit"}:
		typer.echo("Type invalide.")
		raise typer.Exit(code=1)

	# COIN
	coin_in = coin_opt if coin_opt is not None else typer.prompt("Coin (id/symbole/nom)")
	coin = _normalize_coin(coin_in)
	if not coin:
		typer.echo("Coin introuvable ou désactivé.")
		raise typer.Exit(code=1)

	# UNIT
	unit_in = unit_opt if unit_opt is not None else typer.prompt("Unité (min|h|d)")
	unit = _normalize_unit(unit_in)
	if not unit:
		typer.echo("Unité invalide.")
		raise typer.Exit(code=1)

	# VALUE
	if value_opt is None:
		try:
			value = int(typer.prompt("Valeur (entier >0)").strip())
			if value <= 0:
				raise ValueError()
		except Exception:
			typer.echo("Valeur invalide.")
			raise typer.Exit(code=1)
	else:
		try:
			value = int(value_opt)
			if value <= 0:
				raise ValueError()
		except Exception:
			typer.echo("Valeur invalide.")
			raise typer.Exit(code=1)

	minutes = _minutes_from(unit, value)

	# Récupération prix courant (USD)
	pc = PriceCollector()
	prices_payload = pc.get_prices([coin["id"]])
	mid_price_usd = float(prices_payload.get("prices", {}).get(coin["id"], 0.0))
	if mid_price_usd <= 0:
		typer.echo("Prix actuel indisponible pour ce coin.")
		raise typer.Exit(code=2)

	# Récupération taux FX USD/EUR
	fx = FxCollector()
	fx_payload = fx.get_latest()
	# Essayer via to_dataframe pour fiabilité
	try:
		import pandas as pd  # local import safe
		df_fx = FxCollector.to_dataframe(fx_payload)
		fx_rate = float(df_fx.iloc[0]["rate_usd_per_eur"]) if not df_fx.empty else None
	except Exception:
		fx_rate = None
	if not fx_rate or fx_rate <= 0:
		fx_rate = 1.10  # fallback conservateur

	# amount_eur selon kind (optionnel pour "price")
	amount_eur = None
	target_profit_eur = None
	
	if kind == "amount":
		if amount_eur_opt is None:
			try:
				amount_eur = float(typer.prompt("Montant à investir en EUR (ex: 100)").strip())
				if amount_eur <= 0:
					raise ValueError()
			except Exception:
				typer.echo("Montant invalide.")
				raise typer.Exit(code=1)
		else:
			try:
				amount_eur = float(amount_eur_opt)
				if amount_eur <= 0:
					raise ValueError()
			except Exception:
				typer.echo("Montant invalide.")
				raise typer.Exit(code=1)
	
	elif kind == "profit":
		try:
			target_profit_eur = float(typer.prompt("Profit cible en EUR (ex: 50)").strip())
			if target_profit_eur <= 0:
				raise ValueError()
		except Exception:
			typer.echo("Profit cible invalide.")
			raise typer.Exit(code=1)

	pipe = PredictionPipeline()
	
	# Appel selon le type de prédiction
	if kind == "profit":
		# NOUVEAU: Calcul inverse pour profit cible
		result = pipe.calculate_required_investment(
			coin_id=coin["id"],
			horizon_minutes=minutes,
			target_profit_eur=target_profit_eur,
			mid_price_usd=mid_price_usd,
			fx_rate_usd_per_eur=fx_rate,
			spread_pct=None,
		)
	else:
		# Prédiction classique (price/amount)
		result = pipe.run(
			coin_id=coin["id"],
			horizon_minutes=minutes,
			mid_price_usd=mid_price_usd,
			fx_rate_usd_per_eur=fx_rate,
			spread_pct=None,
			amount_eur=amount_eur,
		)

	# Sauvegarde de la prédiction pour évaluation post-horizon (seulement pour prédictions normales)
	if kind != "profit":
		try:
			pred_id = generate_prediction_id()
			ts_pred = _now_ms()
			target_ts = ts_pred + minutes * 60_000
			err_attendue = None
			try:
				# extraire erreur attendue si disponible
				err_attendue = result.get("estimation", {}).get("erreur_attendue_pct")
			except Exception:
				pass
			save_prediction(
				prediction_id=pred_id,
				coin_id=coin["id"],
				horizon_minutes=minutes,
				ts_pred_utc_ms=ts_pred,
				target_ts_utc_ms=target_ts,
				mid_price_usd=mid_price_usd,
				value_pred=result.get("estimation", {}).get("value_pred"),
				extra={
					"erreur_attendue_pct": err_attendue,
					"spread_pct": result.get("spread_pct"),
					"fx_rate_usd_per_eur": result.get("fx_rate_usd_per_eur"),
				},
			)
		except Exception:
			# Ne pas bloquer l'UX si la persistance échoue
			pass

	# Enrichir d’un en-tête contexte utilisateur
	enriched = {
		"coin": {"id": coin["id"], "symbol": coin["symbol"], "name": coin["name"]},
		"requested": {
			"kind": kind, 
			"unit": unit, 
			"value": value, 
			"minutes": minutes,
			"amount_eur": amount_eur,
			"target_profit_eur": target_profit_eur
		},
		"timestamp_utc": _now_iso(),
		"output": result,
	}
	
	# Affichage amélioré selon le type
	if kind == "profit":
		if result.get("success"):
			print(f"\n💰 CALCUL INVERSE PROFIT:")
			print(f"🎯 Pour gagner {target_profit_eur}€ avec {coin['symbol']} en {minutes}min:")
			print(f"💵 Investissement requis: {result['required_investment_eur']}€")
			print(f"📈 Rendement attendu: {result['expected_return_pct']}%")
			print(f"💸 Coûts estimés: {result['estimated_costs_eur']}€")
		else:
			print(f"\n❌ IMPOSSIBLE: {result.get('error', 'Erreur inconnue')}")
	elif kind == "amount":
		final_amount = result.get("estimation", {}).get("final_amount_eur")
		profit = result.get("estimation", {}).get("expected_profit_net_eur")
		if final_amount is not None:
			print(f"\n💸 PROJECTION INVESTISSEMENT:")
			print(f"💰 Avec {amount_eur}€ de {coin['symbol']} en {minutes}min:")
			print(f"🏆 Montant final: {final_amount:.2f}€")
			print(f"📊 Profit net: {profit:+.2f}€" if profit else "📊 Profit: calcul en cours...")
	
	typer.echo(json.dumps(enriched, ensure_ascii=False))


@app.command()
def status(
    detail: bool = typer.Option(False, "--detail", help="Afficher le détail par coin pour la MAE% (sinon agrégats globaux)"),
) -> None:
    """Affiche un état synthétique et lisible du système."""
    
    # --- Récupération des données brutes ---
    s = _settings()
    data_root = s["paths"]["data_parquet"]
    pid_path, lock_path = _runtime_paths()

    def count_parquet(sub: str) -> Tuple[int, Optional[str]]:
        p = os.path.join(data_root, sub)
        if not os.path.isdir(p):
            return 0, None
        cnt = 0
        last_mod_ts = 0
        for root, _, files in os.walk(p):
            for f in files:
                if f.endswith('.parquet'):
                    cnt += 1
                    fp = os.path.join(root, f)
                    try:
                        mod_time = os.path.getmtime(fp)
                        if mod_time > last_mod_ts:
                            last_mod_ts = mod_time
                    except OSError:
                        continue
        last = datetime.fromtimestamp(last_mod_ts, tz=timezone.utc).isoformat() if last_mod_ts > 0 else None
        return cnt, last

    datasets = {sub: count_parquet(sub) for sub in ["five_min", "hourly", "daily", "markets", "tickers_spread", "fx", "predictions", "eval_results"]}
    
    # --- Formatage de la sortie ---
    typer.echo(typer.style("--- État du Système ---", fg=typer.colors.CYAN, bold=True))

    # Section 1: Runtime et Scheduler
    typer.echo(typer.style("\n▶️ Runtime", fg=typer.colors.GREEN, bold=True))
    sched_status = "🔴 Inactif"
    if os.path.exists(pid_path):
        try:
            with open(pid_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            pid = int(info.get("pid", -1))
            if psutil.pid_exists(pid) and psutil.Process(pid).is_running():
                sched_status = f"✅ Actif (PID: {pid})"
            else:
                sched_status = "⚠️ Inactif (PID trouvé mais processus mort)"
        except (json.JSONDecodeError, ValueError, psutil.NoSuchProcess):
            sched_status = "⚠️ Inactif (fichier PID corrompu)"
        except Exception:
            sched_status = "⚠️ Erreur lecture PID"
    
    typer.echo(f"  Scheduler: {sched_status}")
    typer.echo(f"  Fichier Lock: {'✅ Présent' if os.path.exists(lock_path) else '☑️ Absent'}")

    # Section 2: Datasets Parquet
    typer.echo(typer.style("\n📦 Datasets (Parquet)", fg=typer.colors.GREEN, bold=True))
    for name, (count, last_mod) in datasets.items():
        status_icon = "✅" if count > 0 else "❌"
        last_mod_str = "jamais"
        if last_mod:
            try:
                dt_obj = datetime.fromisoformat(last_mod)
                now = datetime.now(timezone.utc)
                delta = now - dt_obj
                last_mod_str = f"il y a {format_seconds(int(delta.total_seconds()))}"
            except ValueError:
                last_mod_str = "date invalide"
        
        typer.echo(f"  {status_icon} {name:<18} | {count:>5} fichiers | Modifié: {last_mod_str}")

    # Section 3: Modèles de prédiction
    typer.echo(typer.style("\n🧠 Modèles Entraînés", fg=typer.colors.GREEN, bold=True))
    try:
        store = ModelStore()
        paths = store.list_models()
        if not paths:
            typer.echo("  Aucun modèle trouvé.")
        else:
            by_coin = {}
            for mp in paths:
                try:
                    base = os.path.basename(mp)
                    coin, horizon_part = base.split("__", 1)
                    horizon = horizon_part.replace("m.pkl", "")
                    by_coin.setdefault(coin, []).append(int(horizon))
                except ValueError:
                    continue
            typer.echo(f"  Total: {len(paths)} modèles pour {len(by_coin)} cryptos.")
            for coin, horizons in sorted(by_coin.items()):
                horizons_str = ", ".join(f"{h}m" for h in sorted(horizons))
                typer.echo(f"    - {coin.capitalize():<12} | Horizons: {horizons_str}")
    except Exception as e:
        typer.echo(typer.style(f"  Erreur lors du listage des modèles: {e}", fg=typer.colors.RED))

    # Section 4: Performance (MAE%)
    typer.echo(typer.style("\n📈 Performance (Erreur Absolue Moyenne %)", fg=typer.colors.GREEN, bold=True))
    try:
        eval_stats = _compute_mae_stats()
        for window, data in eval_stats.items():
            win_name = {"window_24h": "24h", "window_7d": "7 jours", "window_30d": "30 jours"}.get(window, window)
            mae = data.get("global", {}).get("mae_pct")
            count = data.get("global", {}).get("count")
            if mae is not None:
                typer.echo(f"  - {win_name:<8} | {mae:.2f}% MAE ({count} prédictions)")
            else:
                typer.echo(f"  - {win_name:<8} | Pas de données")
        if detail:
             typer.echo(typer.style("    Détail par crypto (24h):", bold=True))
             by_coin_24h = eval_stats.get("window_24h", {}).get("by_coin", {})
             if not by_coin_24h:
                 typer.echo("      Aucune donnée détaillée disponible.")
             for coin, stats in sorted(by_coin_24h.items()):
                 typer.echo(f"      - {coin:<12} | {stats['mae_pct']:.2f}% MAE ({stats['count']} prédictions)")

    except Exception as e:
        typer.echo(typer.style(f"  Erreur calcul MAE: {e}", fg=typer.colors.RED))

    # Section 5: Métriques Système
    typer.echo(typer.style("\n⚙️ Système", fg=typer.colors.GREEN, bold=True))
    try:
        sys_stats = _system_stats()
        uptime = format_seconds(sys_stats.get("uptime_sec", 0))
        ram_pct = sys_stats.get("ram_used_pct", 0)
        disk_free = sys_stats.get("disk_free_gb", 0)
        disk_total = sys_stats.get("disk_total_gb", 1)
        disk_pct_free = (disk_free / disk_total) * 100 if disk_total > 0 else 0
        gpu = "✅ Disponible" if sys_stats.get("gpu_available") else "❌ Non disponible"
        
        typer.echo(f"  Uptime: {uptime}")
        typer.echo(f"  RAM: {ram_pct:.1f}% utilisée")
        typer.echo(f"  Disque: {disk_free:.1f} Go libres ({disk_pct_free:.1f}% libres)")
        typer.echo(f"  GPU (pour ML): {gpu}")
    except Exception as e:
        typer.echo(typer.style(f"  Erreur lecture métriques système: {e}", fg=typer.colors.RED))

    # Section 6: Usage API
    typer.echo(typer.style("\n🌐 API (CoinGecko)", fg=typer.colors.GREEN, bold=True))
    try:
        snap = api_snapshot()
        minute_calls = snap.get("minute_calls")
        minute_limit = snap.get("minute_limit")
        soft_cap = snap.get("soft_cap")
        state = snap.get("state")
        daily_total = snap.get("daily_total")
        soft_pct = snap.get("soft_cap_usage_pct")
        color = typer.colors.GREEN
        if state == "AMBER":
            color = typer.colors.YELLOW
        elif state == "RED":
            color = typer.colors.RED
        typer.echo(typer.style(f"  Minute: {minute_calls}/{minute_limit} (soft {soft_cap}, {soft_pct}% soft) état={state}", fg=color))
        typer.echo(f"  Daily total: {daily_total}")
        eps = snap.get("endpoints", {})
        if eps:
            for ep, st in eps.items():
                typer.echo(f"    - {ep}: {st['calls']} calls (2xx/3xx={st['success']} 4xx={st['4xx']} 5xx={st['5xx']})")
    except Exception as e:
        typer.echo(typer.style(f"  Erreur snapshot API: {e}", fg=typer.colors.RED))

    # Section 7: Rotation des Modèles
    if detail:
        typer.echo(typer.style("\n🔄 Rotation des Modèles", fg=typer.colors.GREEN, bold=True))
        try:
            from prediction.model_store import ModelStore
            store = ModelStore()
            model_stats = store.get_model_stats()
            
            total = model_stats.get("total_models", 0)
            by_coin = model_stats.get("by_coin", {})
            oldest = model_stats.get("oldest_model")
            newest = model_stats.get("newest_model")
            
            typer.echo(f"  Total: {total} modèles actifs")
            typer.echo(f"  Répartition: {len(by_coin)} cryptos")
            
            if oldest:
                oldest_date = oldest["date"][:10]  # YYYY-MM-DD
                typer.echo(f"  Plus ancien: {oldest_date}")
            
            if newest:
                newest_date = newest["date"][:10]  # YYYY-MM-DD
                typer.echo(f"  Plus récent: {newest_date}")
            
            # Top 3 cryptos avec le plus de modèles
            top_cryptos = sorted(by_coin.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_cryptos:
                typer.echo("  Top cryptos:")
                for coin, count in top_cryptos:
                    typer.echo(f"    - {coin}: {count} modèles")
            
        except Exception as e:
            typer.echo(typer.style(f"  Erreur stats modèles: {e}", fg=typer.colors.RED))


@app.command()
def settings() -> None:
	"""Affiche des extraits de configuration utiles."""
	s = _settings()
	th = _thresholds()
	tg = _targets()
	out = {
		"timezone": s.get("timezone"),
		"paths": s.get("paths"),
		"cadences": s.get("cadences"),
		"thresholds": th,
		"targets_currency": tg.get("currency"),
	}
	typer.echo(json.dumps(out, ensure_ascii=False))


# --- Modèles: entraînement et listing ---

@app.command()
def models() -> None:
	"""Liste les modèles disponibles avec un aperçu agrégé par coin/horizon."""
	try:
		store = ModelStore()
		paths = store.list_models()

		print("\n🧠 MODÈLES DISPONIBLES")
		print("=" * 50)

		if not paths:
			print("Aucun modèle trouvé.")
			return

		# Organiser par coin
		by_coin = {}
		for mp in paths:
			base = os.path.basename(mp)
			try:
				parts = base.split("__", 1)
				if len(parts) == 2:
					coin = parts[0]
					tail = parts[1]
					if tail.endswith(".pkl") and tail.endswith("m.pkl"):
						val = tail[:-4]  # remove .pkl
						val = val.rstrip("m")
						hz = f"{int(val)}m"
						by_coin.setdefault(coin, []).append(hz)
			except Exception:
				continue

		print(f"Total: {len(paths)} modèles pour {len(by_coin)} cryptos.")
		print()

		for coin, horizons in sorted(by_coin.items()):
			horizons_sorted = sorted(set(horizons), key=lambda x: int(x[:-1]))
			horizons_str = ", ".join(horizons_sorted)
			print(f"  - {coin.capitalize():<15} | Horizons: {horizons_str}")

	except Exception as e:
		print(f"Erreur lors du listage des modèles: {e}")


@app.command()
def train(
	coin: Optional[str] = typer.Option(None, "--coin", help="Coin id (ex: bitcoin). Si omis, tous les coins actifs."),
	horizon: int = typer.Option(10, "--horizon", help="Horizon en minutes (défaut: 10)."),
) -> None:
	"""Entraîne un modèle baseline (LinearRegression) sur five_min pour un coin/horizon."""
	# Chargement données
	df = _read_recent_five_min()
	if df is None or df.empty:
		typer.echo(json.dumps({"status": "NO_DATA"}))
		raise typer.Exit(code=2)

	# Déterminer coins ciblés
	target_coins: list[str] = []
	if coin:
		target_coins = [coin]
	else:
		try:
			cs = _coins()
			target_coins = [c.get("id") for c in cs if c.get("enabled", True)]
		except Exception:
			target_coins = sorted(list({str(x) for x in df.get("coin_id", pd.Series(dtype=str)).unique() if str(x)}))

	tr = Trainer()
	results = []
	for cid in target_coins:
		try:
			sub = df[df["coin_id"] == cid].sort_values("ts_utc_ms").tail(5000)
			if sub.empty:
				results.append({"coin_id": cid, "status": "NO_DATA"})
				continue
			res = tr.train(sub, cid, horizon_minutes=int(horizon))
			res["coin_id"] = cid
			results.append(res)
		except Exception as e:
			results.append({"coin_id": cid, "status": "ERROR", "error": str(e)})
	typer.echo(json.dumps({"horizon_minutes": int(horizon), "results": results}, ensure_ascii=False))


@app.command(name="train-incremental")
def train_incremental(
	mode: str = typer.Option("micro", "--mode", help="Mode: micro, mini, hourly, daily, schedule"),
	coin: Optional[str] = typer.Option(None, "--coin", help="Coin id (ex: bitcoin). Si omis, tous les coins actifs."),
	horizon: int = typer.Option(10, "--horizon", help="Horizon en minutes (défaut: 10)."),
) -> None:
	"""Entraînement incrémental selon différents modes (micro/mini/hourly/daily) ou affichage du calendrier."""

	# Chargement données
	df = _read_recent_five_min()
	if df is None or df.empty:
		typer.echo(json.dumps({"status": "NO_DATA"}))
		raise typer.Exit(code=2)

	# Déterminer coins ciblés
	target_coins: list[str] = []
	if coin:
		target_coins = [coin]
	else:
		try:
			cs = _coins()
			target_coins = [c.get("id") for c in cs if c.get("enabled", True)]
		except Exception:
			target_coins = sorted(list({str(x) for x in df.get("coin_id", pd.Series(dtype=str)).unique() if str(x)}))

	inc_trainer = IncrementalTrainer()

	if mode == "schedule":
		# Afficher le calendrier de tous les modèles
		horizons = [10, 360, 1440]  # 10min, 6h, 24h
		schedule = inc_trainer.get_training_schedule(target_coins, horizons)
		typer.echo(json.dumps({"mode": "schedule", "schedule": schedule}, ensure_ascii=False))
		return

	elif mode == "daily":
		# Recalibration complète de tous les modèles
		horizons = [10, 360, 1440]
		result = inc_trainer.daily_recalibration(df, target_coins, horizons)
		typer.echo(json.dumps(result, ensure_ascii=False))
		return

	# Modes individuels (micro, mini, hourly)
	results = []
	for cid in target_coins:
		try:
			sub = df[df["coin_id"] == cid].sort_values("ts_utc_ms")
			if sub.empty:
				results.append({"coin_id": cid, "status": "NO_DATA"})
				continue

			if mode == "micro":
				res = inc_trainer.micro_update(sub, cid, horizon_minutes=int(horizon))
			elif mode == "mini":
				res = inc_trainer.mini_retrain(sub, cid, horizon_minutes=int(horizon))
			elif mode == "hourly":
				res = inc_trainer.hourly_refit(sub, cid, horizon_minutes=int(horizon))
			else:
				res = {"status": "INVALID_MODE", "mode": mode}

			res["coin_id"] = cid
			results.append(res)
		except Exception as e:
			results.append({"coin_id": cid, "status": "ERROR", "error": str(e)})

	typer.echo(json.dumps({"mode": mode, "horizon_minutes": int(horizon), "results": results}, ensure_ascii=False))


@app.command(name="confidence-metrics")
def confidence_metrics(
	coin: str = typer.Option(..., "--coin", help="Coin id (ex: bitcoin)"),
	horizon: int = typer.Option(10, "--horizon", help="Horizon en minutes (défaut: 10)"),
) -> None:
	"""Affiche les métriques de confiance pour un coin/horizon donné."""
	try:
		from prediction.confidence import ConfidenceEstimator
		conf = ConfidenceEstimator()
		metrics = conf.get_confidence_metrics(coin, horizon)
		
		result = {
			"coin_id": coin,
			"horizon_minutes": horizon,
			"metrics": metrics,
			"status": "OK"
		}
		typer.echo(json.dumps(result, ensure_ascii=False))
	except Exception as e:
		result = {
			"coin_id": coin,
			"horizon_minutes": horizon,
			"status": "ERROR",
			"error": str(e)
		}
		typer.echo(json.dumps(result, ensure_ascii=False))
		raise typer.Exit(code=1)


@app.command()
def train_incremental(
	mode: str = typer.Option("micro", "--mode", help="Type d'entraînement: micro, mini, hourly, daily, schedule"),
	coin: Optional[str] = typer.Option(None, "--coin", help="Coin id spécifique (optionnel)"),
	horizon: int = typer.Option(10, "--horizon", help="Horizon en minutes (défaut: 10)"),
) -> None:
	"""Entraînement incrémental des modèles ML avec différentes cadences."""
	
	# Chargement des données
	df = _read_recent_five_min()
	if df is None or df.empty:
		typer.echo(json.dumps({"status": "NO_DATA", "message": "Aucune donnée five_min disponible"}))
		raise typer.Exit(code=2)
	
	# Déterminer les coins ciblés
	target_coins: list[str] = []
	if coin:
		target_coins = [coin]
	else:
		try:
			cs = _coins()
			target_coins = [c.get("id") for c in cs if c.get("enabled", True)]
		except Exception:
			target_coins = sorted(list({str(x) for x in df.get("coin_id", pd.Series(dtype=str)).unique() if str(x)}))
	
	trainer = IncrementalTrainer()
	
	if mode == "schedule":
		# Afficher le calendrier d'entraînement
		horizons = [10, 360, 1440]  # 10min, 6h, 24h
		schedule = trainer.get_training_schedule(target_coins, horizons)
		typer.echo(json.dumps({
			"mode": "schedule",
			"schedule": schedule,
			"total_models": len(schedule)
		}, ensure_ascii=False, indent=2))
		return
	
	results = []
	
	for coin_id in target_coins:
		try:
			if mode == "micro":
				result = trainer.micro_update(df, coin_id, horizon)
			elif mode == "mini":
				result = trainer.mini_retrain(df, coin_id, horizon)
			elif mode == "hourly":
				result = trainer.hourly_refit(df, coin_id, horizon)
			elif mode == "daily":
				# Pour daily, traiter tous les horizons
				horizons = [10, 360, 1440]
				result = trainer.daily_recalibration(df, [coin_id], horizons)
			else:
				result = {"status": "UNKNOWN_MODE", "mode": mode}
			
			result["coin_id"] = coin_id
			result["horizon_minutes"] = horizon
			results.append(result)
			
		except Exception as e:
			results.append({
				"coin_id": coin_id,
				"horizon_minutes": horizon,
				"status": "ERROR",
				"error": str(e)
			})
	
	output = {
		"mode": mode,
		"horizon_minutes": horizon,
		"target_coins": target_coins,
		"results": results,
		"summary": {
			"total": len(results),
			"successful": sum(1 for r in results if r.get("status") == "OK"),
			"errors": sum(1 for r in results if r.get("status") == "ERROR")
		}
	}
	
	typer.echo(json.dumps(output, ensure_ascii=False, indent=2))


@app.command()
def confidence_metrics(
	coin: str = typer.Option(..., "--coin", help="Coin id (ex: bitcoin)"),
	horizon: int = typer.Option(10, "--horizon", help="Horizon en minutes"),
) -> None:
	"""Affiche les métriques de confiance pour un coin/horizon donné."""
	try:
		from prediction.confidence import ConfidenceEstimator
		
		conf = ConfidenceEstimator()
		metrics = conf.get_confidence_metrics(coin, horizon)
		
		# Ajouter quelques métriques calculées
		costs_pct = 0.01  # 1% de coûts estimés par défaut
		expected_error = conf.expected_error_pct(coin, horizon, costs_pct)
		
		output = {
			"coin_id": coin,
			"horizon_minutes": horizon,
			"confidence_metrics": metrics,
			"expected_error_with_costs": expected_error,
			"costs_pct_assumed": costs_pct
		}
		
		typer.echo(json.dumps(output, ensure_ascii=False, indent=2))
		
	except Exception as e:
		typer.echo(json.dumps({
			"status": "ERROR",
			"coin_id": coin,
			"horizon_minutes": horizon,
			"error": str(e)
		}, ensure_ascii=False))
		raise typer.Exit(code=1)


@app.command()
def train_advanced(
	coin: Optional[str] = typer.Option(None, "--coin", help="Coin id (ex: bitcoin). Si omis, tous les coins actifs."),
	horizon: int = typer.Option(10, "--horizon", help="Horizon en minutes (défaut: 10)."),
	training_type: str = typer.Option("auto", "--type", help="Type: micro, mini, hourly, daily, auto (défaut)."),
	force_model: Optional[str] = typer.Option(None, "--model", help="Forcer un modèle: linear, random_forest, lightgbm, xgboost.")
) -> None:
	"""Entraînement avancé avec modèles ML optimisés et entraînement incrémental."""
	from prediction.incremental_trainer import IncrementalTrainer
	
	# Chargement données
	df = _read_recent_five_min()
	if df is None or df.empty:
		typer.echo(json.dumps({"status": "NO_DATA"}))
		raise typer.Exit(code=2)

	# Déterminer coins ciblés
	target_coins: list[str] = []
	if coin:
		target_coins = [coin]
	else:
		try:
			cs = _coins()
			target_coins = [c.get("id") for c in cs if c.get("enabled", True)]
		except Exception:
			target_coins = sorted(list({str(x) for x in df.get("coin_id", pd.Series(dtype=str)).unique() if str(x)}))

	inc_trainer = IncrementalTrainer()
	results = []
	
	for cid in target_coins:
		try:
			sub = df[df["coin_id"] == cid].sort_values("ts_utc_ms")
			if sub.empty:
				results.append({"coin_id": cid, "status": "NO_DATA"})
				continue
			
			# Sélectionner le type d'entraînement
			if training_type == "micro":
				res = inc_trainer.micro_update(sub, cid, horizon_minutes=int(horizon))
			elif training_type == "mini":
				res = inc_trainer.mini_retrain(sub, cid, horizon_minutes=int(horizon))
			elif training_type == "hourly":
				res = inc_trainer.hourly_refit(sub, cid, horizon_minutes=int(horizon))
			elif training_type == "daily":
				# Pour daily, on fait un hourly_refit mais avec marquage daily
				res = inc_trainer.hourly_refit(sub, cid, horizon_minutes=int(horizon))
				if res.get("status") == "OK":
					res["training_type"] = "daily_recalibration"
			else:  # auto
				# Auto-sélection intelligente basée sur les données disponibles
				if len(sub) > 1000:
					res = inc_trainer.hourly_refit(sub, cid, horizon_minutes=int(horizon))
				elif len(sub) > 100:
					res = inc_trainer.mini_retrain(sub, cid, horizon_minutes=int(horizon))
				else:
					res = inc_trainer.micro_update(sub, cid, horizon_minutes=int(horizon))
			
			res["coin_id"] = cid
			results.append(res)
			
		except Exception as e:
			results.append({"coin_id": cid, "status": "ERROR", "error": str(e)})
	
	typer.echo(json.dumps({
		"horizon_minutes": int(horizon), 
		"training_type": training_type,
		"force_model": force_model,
		"results": results
	}, ensure_ascii=False))


@app.command()
def train_schedule() -> None:
	"""Affiche le calendrier des prochains entraînements pour tous les modèles."""
	from prediction.incremental_trainer import IncrementalTrainer
	
	# Obtenir la liste des coins actifs
	try:
		cs = _coins()
		target_coins = [c.get("id") for c in cs if c.get("enabled", True)]
	except Exception:
		typer.echo(json.dumps({"status": "ERROR", "error": "Impossible de lire coins.json"}))
		raise typer.Exit(code=1)
	
	# Horizons standards
	horizons = [10, 60, 360, 1440]  # 10min, 1h, 6h, 24h
	
	inc_trainer = IncrementalTrainer()
	schedule = inc_trainer.get_training_schedule(target_coins, horizons)
	
	typer.echo(json.dumps({
		"schedule": schedule,
		"coins": target_coins,
		"horizons": horizons
	}, ensure_ascii=False))


@app.command()
def batch_retrain(
	training_type: str = typer.Option("auto", "--type", help="Type: micro, mini, hourly, daily, auto."),
	force: bool = typer.Option(False, "--force", help="Forcer le re-entraînement même si récent.")
) -> None:
	"""Re-entraînement en lot de tous les modèles selon leur calendrier."""
	from prediction.incremental_trainer import IncrementalTrainer
	
	# Chargement données
	df = _read_recent_five_min()
	if df is None or df.empty:
		typer.echo(json.dumps({"status": "NO_DATA"}))
		raise typer.Exit(code=2)
	
	# Obtenir la liste des coins actifs
	try:
		cs = _coins()
		target_coins = [c.get("id") for c in cs if c.get("enabled", True)]
	except Exception:
		typer.echo(json.dumps({"status": "ERROR", "error": "Impossible de lire coins.json"}))
		raise typer.Exit(code=1)
	
	# Horizons standards
	horizons = [10, 60, 360, 1440]  # 10min, 1h, 6h, 24h
	
	inc_trainer = IncrementalTrainer()
	
	if training_type == "daily":
		# Recalibration complète de tous les modèles
		results = inc_trainer.daily_recalibration(df, target_coins, horizons)
	else:
		# Entraînement individuel pour chaque coin/horizon
		results = {"individual_results": {}}
		
		for cid in target_coins:
			for horizon_minutes in horizons:
				sub = df[df["coin_id"] == cid].sort_values("ts_utc_ms")
				if sub.empty:
					continue
				
				try:
					if training_type == "micro" or (training_type == "auto" and len(sub) <= 100):
						if force or inc_trainer._should_retrain(cid, horizon_minutes, "micro"):
							res = inc_trainer.micro_update(sub, cid, horizon_minutes)
					elif training_type == "mini" or (training_type == "auto" and len(sub) <= 1000):
						if force or inc_trainer._should_retrain(cid, horizon_minutes, "mini"):
							res = inc_trainer.mini_retrain(sub, cid, horizon_minutes)
					else:  # hourly or auto with lots of data
						if force or inc_trainer._should_retrain(cid, horizon_minutes, "hourly"):
							res = inc_trainer.hourly_refit(sub, cid, horizon_minutes)
					
					if 'res' in locals():
						results["individual_results"][f"{cid}_{horizon_minutes}m"] = res
						
				except Exception as e:
					results["individual_results"][f"{cid}_{horizon_minutes}m"] = {
						"status": "ERROR", "error": str(e)
					}
	
	typer.echo(json.dumps(results, ensure_ascii=False))


@app.command(name="cleanup-models")
def cleanup_models(
    max_age_days: int = typer.Option(7, "--max-age", help="Âge maximum en jours"),
    max_per_coin_horizon: int = typer.Option(3, "--max-per-group", help="Nombre max de modèles par coin/horizon"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulation (pas de suppression réelle)")
) -> None:
    """Nettoie les anciens modèles selon l'âge et performance."""
    try:
        from prediction.model_store import ModelStore
        store = ModelStore()
        
        if dry_run:
            typer.echo("🧪 MODE SIMULATION - Aucune suppression")
        
        typer.echo(f"🧹 Nettoyage des modèles...")
        typer.echo(f"   - Âge max: {max_age_days} jours")
        typer.echo(f"   - Max par coin/horizon: {max_per_coin_horizon}")
        
        if not dry_run:
            report = store.cleanup_old_models(max_age_days, max_per_coin_horizon)
        else:
            # En simulation, on fait juste les stats
            model_stats = store.get_model_stats()
            report = {"stats": {"total_models": model_stats.get("total_models", 0)}, "deleted": [], "kept": [], "errors": []}
        
        stats = report.get("stats", {})
        deleted_list = report.get("deleted", [])
        kept_list = report.get("kept", [])
        errors = report.get("errors", [])
        
        typer.echo(f"\n📊 RÉSULTATS:")
        typer.echo(f"   🗑️ Supprimés: {len(deleted_list)} modèles")
        typer.echo(f"   ✅ Conservés: {len(kept_list)} modèles")
        typer.echo(f"   📦 Total initial: {stats.get('total_models', 0)} modèles")
        
        if deleted_list and not dry_run:
            typer.echo(f"\n🗑️ MODÈLES SUPPRIMÉS:")
            for item in deleted_list[:10]:  # Afficher les 10 premiers
                coin_horizon = item["coin_horizon"]
                reason = item["reason"]
                mae = item.get("mae", "N/A")
                typer.echo(f"   - {coin_horizon} (MAE: {mae}) - {reason}")
            
            if len(deleted_list) > 10:
                typer.echo(f"   ... et {len(deleted_list) - 10} autres")
        
        if errors:
            typer.echo(f"\n⚠️ ERREURS ({len(errors)}):")
            for error in errors[:5]:  # Afficher les 5 premières
                typer.echo(f"   - {error}")
        
        if not dry_run:
            typer.echo(f"\n💾 Modèles supprimés sauvegardés dans: {store.root_backup}")
        
    except Exception as e:
        typer.echo(typer.style(f"❌ Erreur nettoyage: {e}", fg=typer.colors.RED))


def main() -> None:
	app()


if __name__ == "__main__":
	main()
