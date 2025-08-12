"""
Historical Collector — Collecte historique massive CoinGecko

Stratégie de collecte optimisée pour récupérer l'historique complet AVANT le passage en API gratuite (1er septembre):
- Daily: 2013-2018 (5 a                # Pause ULTRA-SÉCURISÉE Plan Lite
                print(f"    ⏰ Pause 5s (Plan Lite ULTRA-SÉCURISÉ)...")
                time.sleep(5) pour chaque crypto
- Hourly: 2018-aout 2025 (7 ans) pour chaque crypto  
- 5min: Dernières 24h pour chaque crypto

Utilise les collecteurs existants (OhlcCollector + RangeCollector) de manière intelligente.
Respecte les quotas API et optimise le stockage (300 GB max).
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from collectors.ohlc_collector import OhlcCollector
from collectors.range_collector import RangeCollector
from storage.parquet_writer import ParquetWriter


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class HistoricalCollector:
    """
    Collecteur pour récupération historique massive adaptée au Plan Lite CoinGecko.
    
    PLAN LITE LIMITES:
    - Rate Limit: 500 calls/min
    - Daily data: depuis 2013 ✅
    - Hourly data: depuis 2018 ✅  
    - 5-minutely data: 1 jour seulement ⚠️ (Enterprise requis pour historique)
    """
    
    def __init__(self, writer: Optional[ParquetWriter] = None) -> None:
        self.coins_config = _load_json(os.path.join(CONFIG_DIR, "coins.json"))
        self.writer = writer or ParquetWriter()
        self.ohlc = OhlcCollector(writer=self.writer)
        self.range = RangeCollector(writer=self.writer)
        
        # Dates de disponibilité VRAIES selon Plan Lite (daily depuis 2013 !)
        # Utilisation de market_chart?days=max au lieu de market_chart/range
        self.coin_start_dates = {
            "bitcoin": datetime(2013, 4, 28, tzinfo=timezone.utc),        # Plan Lite: daily depuis 2013 ✅
            "ethereum": datetime(2015, 8, 7, tzinfo=timezone.utc),        # Plan Lite: daily depuis 2013 ✅
            "solana": datetime(2020, 4, 11, tzinfo=timezone.utc),
            "binancecoin": datetime(2017, 9, 16, tzinfo=timezone.utc),    # Plan Lite: daily depuis 2013 ✅
            "ripple": datetime(2013, 8, 4, tzinfo=timezone.utc),          # Plan Lite: daily depuis 2013 ✅
            "cardano": datetime(2017, 10, 18, tzinfo=timezone.utc),       # Plan Lite: daily depuis 2013 ✅
            "avalanche-2": datetime(2020, 9, 22, tzinfo=timezone.utc),
            "chainlink": datetime(2017, 11, 9, tzinfo=timezone.utc),      # Plan Lite: daily depuis 2013 ✅
            "polygon-pos": datetime(2019, 4, 27, tzinfo=timezone.utc),
            "dogecoin": datetime(2013, 12, 15, tzinfo=timezone.utc)       # Plan Lite: daily depuis 2013 ✅
        }
        
        # Fin de collecte historique (avant passage en API gratuite)
        self.daily_end = datetime(2018, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        self.hourly_start = datetime(2018, 1, 1, tzinfo=timezone.utc)
        self.now = datetime.now(timezone.utc)
        self.last_24h = self.now - timedelta(days=1)
    
    def _enabled_coin_ids(self) -> List[str]:
        """Liste des cryptos activées dans la config."""
        coins = self.coins_config.get("coins", [])
        return [c["id"] for c in coins if c.get("enabled", True)]
    
    def _to_ms(self, dt: datetime) -> int:
        """Convertit datetime en timestamp milliseconds."""
        return int(dt.timestamp() * 1000)
    
    def collect_daily_historical(self, coin_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collecte daily depuis date disponibilité pour toutes les cryptos.
        PLAN LITE: market_chart?days=max donne TOUTES les données depuis 2013 !
        TEST CONFIRMÉ: Bitcoin depuis 2013-04-28 avec 4488 points ✅
        """
        ids = coin_ids or self._enabled_coin_ids()
        results = {"collected": [], "errors": [], "total_points": 0}
        
        print(f"📅 Collecte Daily COMPLÈTE avec days=max pour {len(ids)} cryptos (depuis 2013)...")
        import sys
        sys.stdout.flush()  # Forcer l'affichage immédiat
        
        for i, coin_id in enumerate(ids):
            try:
                print(f"  [{i+1}/{len(ids)}] {coin_id} toutes données historiques...")
                sys.stdout.flush()  # Forcer l'affichage immédiat
                
                # Utiliser market_chart?days=max pour TOUTES les données
                data, status = self.range.fetch_historical_all(coin_id)
                
                if not data or "prices" not in data:
                    print(f"    ⚠️ Pas de données pour {coin_id}")
                    continue
                
                # Transformer en DataFrame
                prices = data["prices"]
                volumes = data.get("total_volumes", [])
                
                if not prices:
                    continue
                
                df_data = []
                for j, (ts_ms, price) in enumerate(prices):
                    volume = volumes[j][1] if j < len(volumes) else 0.0
                    df_data.append({
                        "ts_utc_ms": int(ts_ms),
                        "coin_id": coin_id,
                        "o": price, "h": price, "l": price, "c": price,  # Daily OHLC = price
                        "volume": volume,
                        "agg_method": "daily_historical"
                    })
                
                df = pd.DataFrame(df_data)
                
                if not df.empty:
                    # Filtrer pour garder seulement jusqu'à 2018 pour cohérence avec design
                    end_2018_ms = int(self.daily_end.timestamp() * 1000)
                    df = df[df["ts_utc_ms"] <= end_2018_ms]
                    
                    if not df.empty:
                        # Écrire en daily
                        written = self.writer.write("daily", df, dedup_keys=["coin_id","ts_utc_ms"], partition_cols=None)
                        
                        first_date = pd.to_datetime(df["ts_utc_ms"].min(), unit="ms").strftime("%Y-%m-%d")
                        last_date = pd.to_datetime(df["ts_utc_ms"].max(), unit="ms").strftime("%Y-%m-%d")
                        
                        results["collected"].append({
                            "coin_id": coin_id,
                            "timeframe": "daily",
                            "points": len(df),
                            "written": written,
                            "first_date": first_date,
                            "last_date": last_date
                        })
                        results["total_points"] += len(df)
                        print(f"    ✅ {len(df)} points daily ({first_date} → {last_date})")
                        sys.stdout.flush()  # Forcer l'affichage immédiat
                
                # Pause ULTRA-SÉCURISÉE Plan Lite
                print(f"    ⏰ Pause 8s entre cryptos (Plan Lite)...")
                sys.stdout.flush()  # Forcer l'affichage immédiat
                time.sleep(8)
                
            except Exception as e:
                print(f"  ❌ Erreur {coin_id}: {e}")
                results["errors"].append({"coin_id": coin_id, "error": str(e)})
        
        print(f"✅ Daily collecté: {results['total_points']} points pour {len(results['collected'])} cryptos")
        return results
    
    def collect_hourly_historical(self, coin_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collecte hourly 2018-août 2025 pour toutes les cryptos.
        PLAN LITE: Données hourly disponibles depuis 2018 uniquement.
        """
        ids = coin_ids or self._enabled_coin_ids()
        results = {"collected": [], "errors": [], "total_points": 0}
        
        start_ms = self._to_ms(self.hourly_start)  # 2018-01-01
        end_ms = self._to_ms(self.now - timedelta(hours=24))  # Jusqu'à il y a 24h
        
        print(f"⏰ Collecte Hourly historique depuis 2018 pour {len(ids)} cryptos (Plan Lite)...")
        
        for i, coin_id in enumerate(ids):
            try:
                print(f"  [{i+1}/{len(ids)}] {coin_id} hourly...")
                
                # Collecte via RangeCollector 
                df, written = self.range.collect(coin_id, start_ms, end_ms, write=False)  # On traite avant d'écrire
                
                if not df.empty:
                    # Agréger en hourly et écrire dans dataset hourly
                    hourly_df = self._aggregate_to_hourly(df, coin_id)
                    if not hourly_df.empty:
                        written_hourly = self.writer.write("hourly", hourly_df, dedup_keys=["coin_id","ts_utc_ms"], partition_cols=None)
                        results["collected"].append({
                            "coin_id": coin_id,
                            "timeframe": "hourly", 
                            "points": len(hourly_df),
                            "written": written_hourly
                        })
                        results["total_points"] += len(hourly_df)
                        print(f"    ✅ {len(hourly_df)} points hourly collectés")
                
                # Pause pour respecter les quotas Plan Lite (500 calls/min)
                print(f"    ⏰ Pause 3s (respect quota 500/min)...")
                time.sleep(3)
                
            except Exception as e:
                print(f"  ❌ Erreur {coin_id}: {e}")
                results["errors"].append({"coin_id": coin_id, "error": str(e)})
        
        print(f"✅ Hourly collecté: {results['total_points']} points pour {len(results['collected'])} cryptos")
        return results
    
    def collect_5min_recent(self, coin_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collecte 5min dernières 24h pour toutes les cryptos.
        PLAN LITE: Données 5-minutely limitées à 1 jour seulement !
        (Plan Enterprise requis pour données 5min historiques depuis 2018)
        """
        ids = coin_ids or self._enabled_coin_ids()
        results = {"collected": [], "errors": [], "total_points": 0}
        
        start_ms = self._to_ms(self.last_24h)
        end_ms = self._to_ms(self.now)
        
        print(f"⚡ Collecte 5min dernières 24h pour {len(ids)} cryptos (Plan Lite: 1 jour max)...")
        
        for i, coin_id in enumerate(ids):
            try:
                print(f"  [{i+1}/{len(ids)}] {coin_id} 5min...")
                
                # Collecte via RangeCollector (granularité 5min automatique pour 24h)
                df, written = self.range.collect(coin_id, start_ms, end_ms, write=True)  # Direct en five_min
                
                if not df.empty:
                    results["collected"].append({
                        "coin_id": coin_id,
                        "timeframe": "5min",
                        "points": len(df),
                        "written": written
                    })
                    results["total_points"] += len(df)
                    print(f"    ✅ {len(df)} points 5min collectés")
                
                # Pause ULTRA-SÉCURISÉE Plan Lite  
                print(f"    ⏰ Pause 5s (Plan Lite ULTRA-SÉCURISÉ)...")
                time.sleep(5)
                
            except Exception as e:
                print(f"  ❌ Erreur {coin_id}: {e}")
                results["errors"].append({"coin_id": coin_id, "error": str(e)})
        
        print(f"✅ 5min collecté: {results['total_points']} points pour {len(results['collected'])} cryptos")
        return results
    
    def _aggregate_to_daily(self, df, coin_id: str):
        """Agrège des données en daily OHLCV."""
        if df.empty or 'ts_utc_ms' not in df.columns:
            return df
        
        import pandas as pd
        
        # Convertir timestamp en date
        df = df.copy()
        df['date'] = pd.to_datetime(df['ts_utc_ms'], unit='ms', utc=True).dt.date
        
        # Grouper par date et agréger
        daily = df.groupby(['coin_id', 'date']).agg({
            'ts_utc_ms': 'first',  # Premier timestamp de la journée
            'o': 'first',          # Open = premier prix
            'h': 'max',            # High = max prix
            'l': 'min',            # Low = min prix  
            'c': 'last',           # Close = dernier prix
            'volume': 'sum'        # Volume = somme
        }).reset_index()
        
        # Nettoyage
        daily = daily.drop('date', axis=1)
        daily['agg_method'] = 'daily_aggregated'
        
        return daily[["ts_utc_ms","coin_id","o","h","l","c","volume","agg_method"]]
    
    def _aggregate_to_hourly(self, df, coin_id: str):
        """Agrège des données en hourly OHLCV."""
        if df.empty or 'ts_utc_ms' not in df.columns:
            return df
        
        import pandas as pd
        
        # Convertir en heure
        df = df.copy()
        df['hour'] = pd.to_datetime(df['ts_utc_ms'], unit='ms', utc=True).dt.floor('H')
        
        # Grouper par heure et agréger
        hourly = df.groupby(['coin_id', 'hour']).agg({
            'ts_utc_ms': 'first',
            'o': 'first',
            'h': 'max', 
            'l': 'min',
            'c': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # Utiliser le timestamp de l'heure comme référence
        hourly['ts_utc_ms'] = hourly['hour'].astype('int64') // 10**6  # Convert to ms
        hourly = hourly.drop('hour', axis=1)
        hourly['agg_method'] = 'hourly_aggregated'
        
        return hourly[["ts_utc_ms","coin_id","o","h","l","c","volume","agg_method"]]
    
    def collect_all_historical(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Collecte historique complète dans l'ordre optimal.
        MISSION CRITIQUE avant passage API gratuite !
        Si dry_run=True, retourne une simulation.
        """
        if dry_run:
            return self._simulate_collection()
            
        print("🚀 DÉMARRAGE COLLECTE HISTORIQUE MASSIVE")
        print("=" * 60)
        
        total_results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "daily": {},
            "hourly": {},
            "5min": {},
            "summary": {}
        }
        
        try:
            # 1. Daily 2013-2018 (priorité max - base ML)
            print("\n📅 PHASE 1: Daily historique 2013-2018")
            daily_results = self.collect_daily_historical()
            total_results["daily"] = daily_results
            
            # 2. 5min dernières 24h (rapide, pour tests immédiats)
            print("\n⚡ PHASE 2: 5min dernières 24h")  
            min5_results = self.collect_5min_recent()
            total_results["5min"] = min5_results
            
            # 3. Hourly 2018-2025 (gros volume, en dernier)
            print("\n⏰ PHASE 3: Hourly historique 2018-2025")
            hourly_results = self.collect_hourly_historical()
            total_results["hourly"] = hourly_results
            
            # Résumé
            total_points = (daily_results.get("total_points", 0) + 
                          hourly_results.get("total_points", 0) + 
                          min5_results.get("total_points", 0))
            
            total_results["summary"] = {
                "total_points_collected": total_points,
                "daily_cryptos": len(daily_results.get("collected", [])),
                "hourly_cryptos": len(hourly_results.get("collected", [])),
                "5min_cryptos": len(min5_results.get("collected", [])),
                "total_errors": (len(daily_results.get("errors", [])) + 
                               len(hourly_results.get("errors", [])) + 
                               len(min5_results.get("errors", [])))
            }
            
            print("\n" + "=" * 60)
            print("🎯 COLLECTE HISTORIQUE TERMINÉE !")
            print(f"📊 Total points collectés: {total_points:,}")
            print(f"💎 Daily: {len(daily_results.get('collected', []))} cryptos")
            print(f"⏰ Hourly: {len(hourly_results.get('collected', []))} cryptos") 
            print(f"⚡ 5min: {len(min5_results.get('collected', []))} cryptos")
            
            if total_results["summary"]["total_errors"] > 0:
                print(f"⚠️  Erreurs: {total_results['summary']['total_errors']}")
            
        except Exception as e:
            print(f"\n❌ ERREUR CRITIQUE: {e}")
            total_results["critical_error"] = str(e)
        
        total_results["end_time"] = datetime.now(timezone.utc).isoformat()
        return total_results
    
    def _simulate_collection(self) -> Dict[str, Any]:
        """Simulation de la collecte pour validation - ESTIMATIONS CORRIGÉES."""
        coin_ids = self._enabled_coin_ids()
        
        # ESTIMATIONS RÉALISTES CORRIGÉES :
        # Daily : 1 requête par crypto (CoinGecko retourne tous les jours 2013-2018 en une fois)
        daily_requests = len(coin_ids)  # 1 requête/crypto
        daily_points = len(coin_ids) * 5 * 365  # 5 ans * 365 jours de données
        
        # Hourly : 1 requête par crypto (CoinGecko retourne toutes les heures 2018-2025 en une fois)
        hourly_requests = len(coin_ids)  # 1 requête/crypto
        hourly_points = len(coin_ids) * 7 * 365 * 24  # 7 ans de données horaires
        
        # 5min : 1 requête par crypto (dernières 24h)
        fivemin_requests = len(coin_ids)  # 1 requête/crypto
        fivemin_points = len(coin_ids) * 24 * 12  # 24h * 12 points/heure
        
        total_requests = daily_requests + hourly_requests + fivemin_requests  # ~30 requêtes total !
        total_points = daily_points + hourly_points + fivemin_points
        
        # Durée réaliste : requêtes + pauses quota
        estimated_minutes = (total_requests * 0.2) / 60 + (total_requests * 0.2)  # requêtes + pauses
        
        return {
            "summary": {
                "total_points_collected": total_points,
                "total_duration_minutes": estimated_minutes,
                "estimated_size_mb": total_points * 0.001,
                "simulation": True,
                "daily_estimated": daily_points,
                "hourly_estimated": hourly_points,
                "5min_estimated": fivemin_points,
                "coins_count": len(coin_ids),
                "total_api_requests": total_requests,
                "estimated_duration_realistic": f"{estimated_minutes:.1f} minutes (~{estimated_minutes/60:.1f}h)"
            }
        }


def main() -> None:
    """Test de collecte historique."""
    collector = HistoricalCollector()
    
    # Test sur Bitcoin uniquement
    results = collector.collect_5min_recent(["bitcoin"])
    print(f"Test 5min Bitcoin: {results}")


if __name__ == "__main__":
    main()
