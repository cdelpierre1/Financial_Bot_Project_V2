"""
Scheduler — Orchestration des collecteurs et maintenance

Caractéristiques:
- Planifie l’exécution périodique des collecteurs (price, markets, chart, tickers, fx) selon settings.json
- ThreadPool minimal, timers récurrents, arrêt gracieux via Event
- Maintenance: rétention/compaction occasionnelles via ParquetWriter

Note: MVP sans APScheduler; simple, lisible et suffisant pour l’usage local.
"""

from __future__ import annotations

import os
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

import pandas as pd

from collectors.price_collector import PriceCollector
from collectors.markets_collector import MarketsCollector
from collectors.chart_collector import ChartCollector
from collectors.tickers_collector import TickersCollector
from collectors.fx_collector import FxCollector
from collectors.ohlc_collector import OhlcCollector
from collectors.range_collector import RangeCollector
from collectors.historical_collector import HistoricalCollector
from storage.parquet_writer import ParquetWriter
from prediction.evaluation import EvaluationJob
from ops.anti_sleep_win import AntiSleepGuard
from prediction.trainer import Trainer
from prediction.prediction_tracker import PredictionTracker
from ops.api_usage import api_snapshot


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _settings() -> Dict[str, Any]:
	return _load_json(os.path.join(CONFIG_DIR, "settings.json"))


class _PeriodicTask:
	def __init__(self, name: str, interval_sec: int, fn: Callable[[], None], initial_delay_sec: int = 0) -> None:
		self.name = name
		self.interval_sec = max(1, int(interval_sec))
		self.initial_delay_sec = max(0, int(initial_delay_sec))
		self.fn = fn
		self._stop = threading.Event()
		self._thread = threading.Thread(target=self._run, name=f"Task-{name}", daemon=True)

	def start(self) -> None:
		self._thread.start()

	def stop(self) -> None:
		self._stop.set()
		self._thread.join(timeout=5)

	def _run(self) -> None:
		# Délai initial avant la première exécution
		next_run = time.time() + self.initial_delay_sec
		while not self._stop.is_set():
			now = time.time()
			if now >= next_run:
				try:
					self.fn()
				except RuntimeError as e:
					# Rate limiter ou API throttling - attendre un peu plus longtemps
					print(f"Task {self.name} throttled: {e}, retrying in {self.interval_sec + 30}s")
					next_run = now + self.interval_sec + 30  # Attendre 30s de plus
					continue
				except Exception as e:
					# Autres erreurs - log et continuer
					print(f"Task {self.name} error: {e}")
				next_run = now + self.interval_sec
			time.sleep(0.2)


class Scheduler:
	def __init__(self) -> None:
		self.s = _settings()
		self.writer = ParquetWriter()
		# Collecteurs
		self.price = PriceCollector()
		self.markets = MarketsCollector(writer=self.writer)
		self.chart = ChartCollector(writer=self.writer)
		self.tickers = TickersCollector(writer=self.writer)
		self.fx = FxCollector(writer=self.writer)
		self.ohlc = OhlcCollector(writer=self.writer)
		self.range = RangeCollector(writer=self.writer)
		self.historical = HistoricalCollector(writer=self.writer)
		self.eval = EvaluationJob()
		self.trainer = Trainer(verbose=False)  # Mode silencieux pour les updates périodiques
		self.prediction_tracker = PredictionTracker()  # Gestionnaire des prédictions CLI
		# Anti-sommeil (Windows) si activé
		feat = self.s.get("features", {})
		self._antisleep = AntiSleepGuard(enabled=bool(feat.get("anti_sleep_windows", False)))
		# Tâches périodiques normales
		cad = self.s.get("cadences", {}).get("collectors", {})
		self.tasks = [
			_PeriodicTask("price", cad.get("simple_price_pull_sec", 10), self._do_price),
			_PeriodicTask("markets", cad.get("coins_markets_pull_sec", 60), self._do_markets),
			_PeriodicTask("chart", cad.get("market_chart_pull_sec", 300), self._do_chart),
			_PeriodicTask("tickers", cad.get("tickers_pull_sec", 600), self._do_tickers),
			_PeriodicTask("fx", cad.get("fx_pull_sec", 3600), self._do_fx),
			_PeriodicTask("maintenance", 1800, self._maintenance),  # 30 min
			_PeriodicTask("evaluation", cad.get("evaluation_pull_sec", 120), self._evaluation),
		]
		
		# Tâche de vérification des prédictions CLI (toutes les 2 minutes)
		self.tasks.append(_PeriodicTask("prediction_verification", 120, self._do_prediction_verification, initial_delay_sec=30))
		
		# Tâche d'entraînement périodique (mini retrain 10 min par défaut)
		try:
			train_cad = self.s.get("cadences", {}).get("training", {})
			mini_min = int(train_cad.get("mini_retrain_min", 10))
			# Délai initial de 2 minutes pour laisser le bot se stabiliser
			self.tasks.append(_PeriodicTask("training", max(60, mini_min * 60), self._do_training, initial_delay_sec=120))
		except Exception:
			pass
		
		# Tâche de collecte historique (une seule fois au démarrage si pas déjà fait)
		self._historical_collected = False
		self.tasks.append(_PeriodicTask("historical_once", 300, self._do_historical_once))  # Vérifier toutes les 5 min si pas fait
		
		# Tâches de maintenance historique quotidienne
		self.tasks.append(_PeriodicTask("ohlc_daily", 21600, self._do_ohlc_daily))  # 6h
		self.tasks.append(_PeriodicTask("range_validation", 43200, self._do_range_validation))  # 12h
		
		# Tâche de nettoyage des modèles (quotidienne)
		self.tasks.append(_PeriodicTask("model_cleanup", 86400, self._do_model_cleanup, initial_delay_sec=3600))  # 24h, délai 1h

		# Charger la liste de coins depuis la config
		try:
			coins_cfg = _load_json(os.path.join(CONFIG_DIR, "coins.json"))
			self._coins = [c.get("id") for c in coins_cfg.get("coins", []) if c.get("enabled", True)]
		except Exception:
			self._coins = []

	# --- Actions ---
	def _do_price(self) -> None:
		try:
			self.price.get_prices()
		except Exception:
			pass

	def _do_markets(self) -> None:
		try:
			self.markets.collect()
		except Exception:
			pass

	def _do_chart(self) -> None:
		try:
			self.chart.collect()
		except Exception:
			pass

	def _do_tickers(self) -> None:
		try:
			self.tickers.collect()
		except Exception:
			pass

	def _do_fx(self) -> None:
		try:
			self.fx.collect()
		except Exception:
			pass

	def _do_historical_once(self) -> None:
		"""Collecte historique massive une seule fois au démarrage."""
		if self._historical_collected:
			return
		
		# VÉRIFIER SI ON A DÉJÀ DES DONNÉES !
		from storage.parquet_writer import ParquetWriter
		writer = ParquetWriter()
		data_path = writer.settings["paths"]["data_parquet"]
		
		# Checker si des données existent déjà
		datasets = ["five_min", "hourly", "daily"]
		has_data = False
		for dataset in datasets:
			dataset_path = os.path.join(data_path, dataset)
			if os.path.exists(dataset_path) and os.listdir(dataset_path):
				has_data = True
				break
		
		if has_data:
			print("📊 DONNÉES HISTORIQUES DÉJÀ PRÉSENTES - Skip collecte automatique")
			self._historical_collected = True  # Marquer comme fait
			return
		
		try:
			print("🚀 Démarrage collecte historique massive...")
			results = self.historical.collect_all_historical()
			
			# Marquer comme fait
			self._historical_collected = True
			
			# Log résultats
			summary = results.get("summary", {})
			total_points = summary.get("total_points_collected", 0)
			print(f"✅ Collecte historique terminée: {total_points:,} points collectés")
			
		except Exception as e:
			print(f"❌ Erreur collecte historique: {e}")
			# Ne pas marquer comme fait pour réessayer au prochain démarrage

	def _do_ohlc_daily(self) -> None:
		"""Collecte OHLC quotidienne pour maintenir l'historique."""
		try:
			print(f"📊 Collecte OHLC quotidienne...")
			# Utiliser la liste des coins configurés
			coin_ids = ["bitcoin", "ethereum", "solana", "binancecoin", "cardano"]
			self.ohlc.collect_daily_batch(coin_ids, days=1)
			print(f"✅ OHLC quotidien terminé")
		except Exception as e:
			print(f"❌ Erreur OHLC quotidien: {e}")

	def _do_range_validation(self) -> None:
		"""Collecte de ranges pour validation croisée."""
		try:
			print(f"📈 Collecte ranges validation...")
			# Utiliser range des dernières 24h
			import time
			now_ms = int(time.time() * 1000)
			frm_ms = now_ms - (24 * 3600 * 1000)  # 24h ago
			coin_ids = ["bitcoin", "ethereum", "solana"]
			self.range.collect_validation_ranges(coin_ids, frm_ms, now_ms)
			print(f"✅ Collecte ranges terminée")
		except Exception as e:
			print(f"❌ Erreur collecte ranges: {e}")

	def _maintenance(self) -> None:
		try:
			# Rétention optimisée selon nouvelle stratégie 300 GB
			self.writer.enforce_retention("five_min", timeframe="five_min")  # 30 jours
			self.writer.enforce_retention("hourly", timeframe="hourly")      # 2 ans  
			self.writer.enforce_retention("daily", timeframe="daily")        # 100 ans (léger)
			self.writer.enforce_retention("markets", timeframe="markets")    # Config
			self.writer.enforce_retention("tickers_spread", timeframe="tickers_spread")
			self.writer.enforce_retention("fx", timeframe="hourly")
			self.writer.enforce_retention("predictions", timeframe="predictions")
			self.writer.enforce_retention("eval_results", timeframe="eval_results")
			
			# Compaction optimisée
			today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
			for ds in ["five_min", "hourly", "daily", "markets", "tickers_spread", "fx", "predictions", "eval_results"]:
				self.writer.compact_partition(ds, today)
				
			# Rotation intelligente des données anciennes (si espace > 250 GB)
			self._auto_data_rotation()
			
		except Exception:
			pass

	def _auto_data_rotation(self) -> None:
		"""Rotation automatique si approche de la limite 300 GB."""
		try:
			import shutil
			
			# Vérifier espace utilisé
			data_path = self.s["paths"]["data_parquet"]
			if os.path.exists(data_path):
				total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
								for dirpath, dirnames, filenames in os.walk(data_path)
								for filename in filenames)
				size_gb = total_size / (1024**3)
				
				# Si > 250 GB, commencer rotation (marge sécurité)
				if size_gb > 250:
					print(f"⚠️  Rotation données: {size_gb:.1f} GB utilisés")
					# Supprimer les plus anciens five_min (garder 7 derniers jours)
					self.writer.enforce_retention("five_min", days=7)
					# Réduire hourly à 1 an
					self.writer.enforce_retention("hourly", days=365)
		except Exception:
			pass

	def _evaluation(self) -> None:
		try:
			self.eval.evaluate_pending()
		except Exception:
			pass

	def _read_recent_five_min(self) -> Optional[pd.DataFrame]:
		"""Lecture robuste du dataset five_min, retour DataFrame ou None."""
		try:
			root = self.s["paths"]["data_parquet"]
			path = os.path.join(root, "five_min")
			if not os.path.isdir(path):
				return None
			try:
				return pd.read_parquet(path)
			except Exception:
				frames: list[pd.DataFrame] = []
				for r, _, files in os.walk(path):
					for f in files:
						if f.endswith(".parquet"):
							fp = os.path.join(r, f)
							try:
								frames.append(pd.read_parquet(fp))
							except Exception:
								pass
				if frames:
					return pd.concat(frames, ignore_index=True)
				return None
		except Exception:
			return None

	def _do_training(self) -> None:
		"""Mini retrain: entraîne un modèle +10 min pour chaque coin actif sur les dernières données."""
		try:
			df = self._read_recent_five_min()
			if df is None or df.empty:
				return
			for cid in self._coins or []:
				try:
					# Garder un sous-ensemble récent par coin (par sécurité)
					sub = df[df["coin_id"] == cid].sort_values("ts_utc_ms").tail(2000)
					if sub.empty:
						continue
					self.trainer.train(sub, cid, horizon_minutes=10)
				except Exception:
					# Continuer avec d'autres coins même si un échoue
					pass
		except Exception:
			pass

	def _do_prediction_verification(self) -> None:
		"""Vérifie les prédictions CLI en attente et met à jour les modèles si nécessaire."""
		try:
			pending_predictions = self.prediction_tracker.check_pending_predictions()
			
			for prediction in pending_predictions:
				try:
					self.prediction_tracker.verify_prediction(prediction)
				except Exception:
					# Continuer avec d'autres prédictions même si une échoue
					pass
					
		except Exception:
			pass

	def _do_model_cleanup(self) -> None:
		"""Nettoyage automatique des anciens modèles."""
		try:
			from prediction.model_store import ModelStore
			store = ModelStore()
			
			# Paramètres de nettoyage (configurables)
			max_age_days = 7  # Supprimer modèles > 7 jours
			max_models_per_coin_horizon = 3  # Garder 3 meilleurs par coin/horizon
			
			print(f"🧹 Nettoyage automatique des modèles...")
			report = store.cleanup_old_models(
				max_age_days=max_age_days,
				max_models_per_coin_horizon=max_models_per_coin_horizon
			)
			
			stats = report.get("stats", {})
			deleted = stats.get("total_deleted", 0)
			kept = stats.get("total_kept", 0)
			
			print(f"   🗑️ Supprimés: {deleted} modèles")
			print(f"   ✅ Conservés: {kept} modèles")
			
			if report.get("errors"):
				print(f"   ⚠️ Erreurs: {len(report['errors'])}")
			
		except Exception as e:
			print(f"⚠️ Erreur nettoyage modèles: {e}")

	# --- Cycle de vie ---
	def start(self) -> None:
		# Activer anti-sommeil best-effort
		try:
			self._antisleep.start()
		except Exception:
			pass
		for t in self.tasks:
			t.start()

	def stop(self) -> None:
		for t in self.tasks:
			t.stop()
		try:
			self._antisleep.stop()
		except Exception:
			pass


def main() -> None:
	sch = Scheduler()
	sch.start()
	try:
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		pass
	finally:
		sch.stop()


if __name__ == "__main__":
	main()

