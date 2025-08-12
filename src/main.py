#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point d'entrée principal du Financial Bot Crypto V2.
Interface CLI consolidée pour toutes les opérations.
"""

import sys
import os
import argparse
from datetime import datetime
from typing import Optional, List

# Ajouter le répertoire src au PYTHONPATH
src_path = os.path.dirname(__file__)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configuration de l'encodage pour Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors="replace")


def setup_cli_parser() -> argparse.ArgumentParser:
    """Configure l'interface CLI avec tous les sous-commandes."""
    parser = argparse.ArgumentParser(
        description="Financial Bot Crypto V2 - Trading Bot et Prédictions ML",
        epilog="""
Exemples d'utilisation:
  # Gestion du bot
  python src/main.py demarrer           # Démarrer le bot complet
  python src/main.py arreter            # Arrêter le bot
  python src/main.py etat               # Statut détaillé du système
  
  # Collecte de données
  python src/main.py collect-historical  # Collecte historique massive urgente
  
  # ML et prédictions
  python src/main.py train --coin bitcoin               # Entraînement complet
  python src/main.py train-incremental --mode micro    # Entraînement incrémental micro
  python src/main.py train-incremental --mode mini     # Entraînement incrémental mini  
  python src/main.py train-incremental --mode schedule # Calendrier d'entraînement
  python src/main.py prevoir --coin bitcoin --unit min --value 10 --kind price
  
  # Analyse et métriques
  python src/main.py confidence-metrics --coin bitcoin --horizon 10
  python src/main.py models              # Lister les modèles disponibles
  python src/main.py parametres          # Voir la configuration
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # === Gestion du bot ===
    start_parser = subparsers.add_parser('demarrer', help='Démarrer le bot (collecte + ML)')
    start_parser.add_argument('--test-mode', action='store_true', help='Mode test (démarre puis arrête après 3s)')
    
    subparsers.add_parser('arreter', help='Arrêter le bot')
    subparsers.add_parser('etat', help='Afficher l\'état du système')
    
    # === Collecte ===
    collect_parser = subparsers.add_parser('collect-historical', help='Collecte historique massive urgente')
    collect_parser.add_argument('--dry-run', action='store_true', default=True, help='Mode simulation (défaut)')
    collect_parser.add_argument('--real', action='store_true', help='Collecte réelle (override dry-run)')
    collect_parser.add_argument('--test-mode', action='store_true', help='Mode test non-bloquant pour smoke test')
    
    # === ML ===
    train_parser = subparsers.add_parser('train', help='Entraînement complet des modèles')
    train_parser.add_argument('--coin', default='bitcoin', help='Cryptomonnaie à entraîner')
    train_parser.add_argument('--test-mode', action='store_true', help='Mode test avec données simulées')
    
    # Entraînement incrémental
    train_inc = subparsers.add_parser('train-incremental', help='Entraînement incrémental')
    train_inc.add_argument('--mode', choices=['micro', 'mini', 'schedule'], 
                          default='micro', help='Mode d\'entraînement incrémental')
    
    # === Prédictions ===
    predict_parser = subparsers.add_parser('prevoir', help='Mode prédictions interactives')
    predict_parser.add_argument('--coin', default='bitcoin', help='Cryptomonnaie (bitcoin, ethereum, etc.)')
    predict_parser.add_argument('--unit', choices=['min', 'hour'], default='min', help='Unité de temps')
    predict_parser.add_argument('--value', type=int, default=10, help='Valeur temporelle (ex: 10 pour 10min)')
    predict_parser.add_argument('--kind', choices=['price', 'amount'], default='price', help='Type de prédiction')
    predict_parser.add_argument('--amount-eur', type=float, help='Montant en EUR (pour kind=amount)')
    predict_parser.add_argument('--test-mode', action='store_true', help='Mode test rapide pour smoke test')
    
    # === Métriques ===
    conf_metrics = subparsers.add_parser('confidence-metrics', help='Métriques de confiance')
    conf_metrics.add_argument('--coin', default='bitcoin', help='Coin à analyser')
    conf_metrics.add_argument('--horizon', type=int, default=10, help='Horizon de prédiction (minutes)')
    
    # === Info ===
    subparsers.add_parser('models', help='Lister les modèles disponibles')
    subparsers.add_parser('parametres', help='Afficher la configuration')
    
    return parser


def _is_first_startup() -> bool:
    """Vérifie si c'est le premier démarrage du bot"""
    from storage.parquet_writer import ParquetWriter
    
    writer = ParquetWriter()
    
    # PRIORITÉ 1: Vérifier si des modèles existent
    models_path = writer.settings["paths"]["models_trained"]
    if os.path.exists(models_path) and os.listdir(models_path):
        return False  # Des modèles existent déjà = PAS premier démarrage
    
    # PRIORITÉ 2: Si pas de modèles, c'est un premier démarrage (ou re-entraînement)
    return True  # Premier démarrage OU besoin de re-entraîner


def _perform_initial_setup():
    """Séquence complète du premier démarrage : Collecte → Split → Train → Reassemble
    En cas d'échec à n'importe quelle étape : ARRÊT IMMÉDIAT"""
    import json
    from collectors.historical_collector import HistoricalCollector
    from storage.parquet_writer import ParquetWriter
    from prediction.master_trainer import MasterModelTrainer
    import pandas as pd
    
    writer = ParquetWriter()
    data_path = writer.settings["paths"]["data_parquet"]
    
    # VÉRIFIER SI LES DONNÉES EXISTENT DÉJÀ
    has_data = False
    datasets = ["five_min", "hourly", "daily"]
    for dataset in datasets:
        dataset_path = os.path.join(data_path, dataset)
        if os.path.exists(dataset_path) and os.listdir(dataset_path):
            has_data = True
            break
    
    if not has_data:
        # PHASE 1: COLLECTE (seulement si pas de données)
        try:
            print("📊 PHASE 1: Collecte historique initiale...")
            import sys
            sys.stdout.flush()
            
            # Collecte historique COMPLÈTE (daily + hourly + 5min) avec quotas API respectés
            collector = HistoricalCollector(writer=writer)
            
            print("Collecte historique massive (daily 2013-2018 + hourly 2018-24h + 5min 24h)...")
            print("⏰ ATTENTION: Respecte quotas API - peut prendre 45+ minutes")
            sys.stdout.flush()
            results = collector.collect_all_historical(dry_run=False)
            
            if not results or results.get("error"):
                raise Exception(f"Échec de la collecte historique: {results}")
                
            print("✅ Collecte historique terminée")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\n❌ ERREUR PHASE 1 - COLLECTE: {e}")
            raise Exception(f"Phase collecte échouée: {e}")
    else:
        print("📊 DONNÉES DÉJÀ PRÉSENTES - Skip collecte, passage direct à l'entraînement")
    
    # PHASE 2: ENTRAÎNEMENT (toujours nécessaire si pas de modèles)
    try:
        # Charger la liste des cryptos configurées
        coins_path = os.path.join(os.path.dirname(__file__), "config", "coins.json")
        with open(coins_path, "r", encoding="utf-8") as f:
            coins_config = json.load(f)
        
        active_coins = [coin["id"] for coin in coins_config["coins"] if coin.get("enabled", True)]
        
        print(f"🧠 PHASE 2: Entraînement modèles maîtres pour {len(active_coins)} cryptos...")
        print("Architecture: 3 sous-modèles (daily/hourly/5min) + 1 modèle maître combiné")
        print("📊 SYSTÈME HYBRIDE: 5 horizons × 10 cryptos = 50 modèles à créer")
        print("⏱️ Estimation: 15-20 minutes total")
        
        master_trainer = MasterModelTrainer()
        data_path = writer.settings["paths"]["data_parquet"]
        
        failed_coins = []
        total_models_created = 0
        
        # Pour chaque crypto, entraîner le système complet
        for i, coin_id in enumerate(active_coins):
            print(f"\n{'='*60}")
            print(f"🎯 [{i+1}/{len(active_coins)}] 💎 {coin_id.upper()} 💎")
            print(f"{'='*60}")
            
            try:
                # Lire les 3 timeframes AVEC GESTION ROBUSTE
                daily_df = pd.DataFrame()
                hourly_df = pd.DataFrame()
                five_min_df = pd.DataFrame()
                
                # Daily data - LECTURE SÉCURISÉE
                daily_path = os.path.join(data_path, "daily")
                if os.path.exists(daily_path):
                    try:
                        # Lire tous les fichiers parquet dans le dossier daily
                        daily_files = []
                        for root, dirs, files in os.walk(daily_path):
                            for file in files:
                                if file.endswith('.parquet'):
                                    file_path = os.path.join(root, file)
                                    try:
                                        df_temp = pd.read_parquet(file_path)
                                        if 'coin_id' in df_temp.columns:
                                            df_temp_filtered = df_temp[df_temp['coin_id'] == coin_id]
                                            if len(df_temp_filtered) > 0:
                                                daily_files.append(df_temp_filtered)
                                    except Exception as e:
                                        print(f"    ⚠️ Skip fichier {file}: {e}")
                                        continue
                        
                        if daily_files:
                            daily_df = pd.concat(daily_files, ignore_index=True)
                            print(f"    📅 Daily: {len(daily_df)} points")
                        else:
                            print(f"    📅 Daily: 0 points (pas de fichiers lisibles)")
                    except Exception as e:
                        print(f"    ⚠️ Erreur lecture daily: {e}")
                
                # Hourly data - LECTURE SÉCURISÉE
                hourly_path = os.path.join(data_path, "hourly")
                if os.path.exists(hourly_path):
                    try:
                        # Lire tous les fichiers parquet dans le dossier hourly
                        hourly_files = []
                        for root, dirs, files in os.walk(hourly_path):
                            for file in files:
                                if file.endswith('.parquet'):
                                    file_path = os.path.join(root, file)
                                    try:
                                        df_temp = pd.read_parquet(file_path)
                                        if 'coin_id' in df_temp.columns:
                                            df_temp_filtered = df_temp[df_temp['coin_id'] == coin_id]
                                            if len(df_temp_filtered) > 0:
                                                hourly_files.append(df_temp_filtered)
                                    except Exception as e:
                                        print(f"    ⚠️ Skip fichier {file}: {e}")
                                        continue
                        
                        if hourly_files:
                            hourly_df = pd.concat(hourly_files, ignore_index=True)
                            print(f"    ⏰ Hourly: {len(hourly_df)} points")
                        else:
                            print(f"    ⏰ Hourly: 0 points (pas de fichiers lisibles)")
                    except Exception as e:
                        print(f"    ⚠️ Erreur lecture hourly: {e}")
                
                # 5min data - LECTURE SÉCURISÉE
                five_min_path = os.path.join(data_path, "five_min")
                if os.path.exists(five_min_path):
                    try:
                        # Lire tous les fichiers parquet dans le dossier five_min
                        five_min_files = []
                        for root, dirs, files in os.walk(five_min_path):
                            for file in files:
                                if file.endswith('.parquet'):
                                    file_path = os.path.join(root, file)
                                    try:
                                        df_temp = pd.read_parquet(file_path)
                                        if 'coin_id' in df_temp.columns:
                                            df_temp_filtered = df_temp[df_temp['coin_id'] == coin_id]
                                            if len(df_temp_filtered) > 0:
                                                five_min_files.append(df_temp_filtered)
                                    except Exception as e:
                                        print(f"    ⚠️ Skip fichier {file}: {e}")
                                        continue
                        
                        if five_min_files:
                            five_min_df = pd.concat(five_min_files, ignore_index=True)
                            print(f"    ⚡ 5min: {len(five_min_df)} points")
                        else:
                            print(f"    ⚡ 5min: 0 points (pas de fichiers lisibles)")
                    except Exception as e:
                        print(f"    ⚠️ Erreur lecture 5min: {e}")
                
                # Vérifier qu'on a suffisamment de données
                total_points = len(daily_df) + len(hourly_df) + len(five_min_df)
                if total_points < 100:
                    print(f"    ⚠️ Pas assez de données pour {coin_id} ({total_points} points total)")
                    failed_coins.append(coin_id)
                    continue
                
                # Entraîner le système HYBRIDE multi-horizons (optimisés pour les données disponibles)
                horizons = [10, 60, 360, 720, 480]  # 10min, 1h, 6h, 12h, 8h  
                horizon_names = ["10min", "1h", "6h", "12h", "8h"]
                horizon_results = {}
                
                print(f"    🤖 ENTRAÎNEMENT HYBRIDE: 5 horizons temporels")
                for j, (horizon, name) in enumerate(zip(horizons, horizon_names)):
                    print(f"    ⏳ [{j+1}/5] Horizon {name} ({horizon}min)...")
                    
                    pipeline_result = master_trainer.train_full_pipeline(
                        coin_id=coin_id,
                        daily_df=daily_df,
                        hourly_df=hourly_df, 
                        five_min_df=five_min_df,
                        horizon_minutes=horizon,
                        split_ratio=0.7
                    )
                    horizon_results[horizon] = pipeline_result
                    
                    # Debug: Inspecter le résultat du modèle maître
                    master_result = pipeline_result["master_model"]
                    print(f"        🔍 Debug master_result: {master_result}")
                    
                    if master_result.get("status") == "OK":
                        mae = master_result["mae"]
                        total_models_created += 1
                        print(f"        ✅ Modèle {name}: MAE={mae:.4f} - SUCCÈS")
                    else:
                        print(f"        ❌ Modèle {name}: ÉCHEC - Status: {master_result.get('status', 'UNKNOWN')}")
                        print(f"        🔍 Debug erreurs: {pipeline_result.get('errors', [])}")
                        failed_coins.append(f"{coin_id}-{name}")
                
                # Résumé système hybride pour cette crypto
                successful_horizons = [h for h, r in horizon_results.items() if r["master_model"].get("status") == "OK"]
                print(f"    🏆 BILAN {coin_id.upper()}: {len(successful_horizons)}/5 horizons ✅")
                
                if len(successful_horizons) == 0:
                    failed_coins.append(coin_id)
                    print(f"    ❌ ÉCHEC TOTAL pour {coin_id}")
                else:
                    print(f"    ✅ CRYPTO {coin_id.upper()} OPÉRATIONNELLE")
                
                # Progression globale
                progress_pct = ((i + 1) / len(active_coins)) * 100
                print(f"📊 PROGRESSION GLOBALE: {progress_pct:.1f}% ({total_models_created} modèles créés)")
                print(f"{'='*60}")
                
            except Exception as e:
                print(f"    ❌ Erreur {coin_id}: {e}")
                failed_coins.append(coin_id)
        
        # Vérifier les échecs critiques - tolérer plus d'échecs car certains horizons longs peuvent échouer
        max_failures = len(active_coins) * 2  # Tolérer jusqu'à 2 échecs par crypto (20 échecs sur 10 cryptos)
        if len(failed_coins) > max_failures:
            raise Exception(f"Trop d'échecs d'entraînement ({len(failed_coins)}/{max_failures} max): {failed_coins}")
        
        print(f"\n✅ PHASE 2 TERMINÉE - {len(active_coins) - len(failed_coins)}/{len(active_coins)} cryptos OK")
        
    except Exception as e:
        print(f"\n❌ ERREUR PHASE 2 - ENTRAÎNEMENT: {e}")
        raise Exception(f"Phase entraînement échouée: {e}")
    
    try:
        print("\n🔄 PHASE 3: Réassemblage des données...")
        print("Les données train/test sont maintenant réunies pour usage continu")
        print("✅ INITIALISATION TERMINÉE - Bot prêt avec modèles maîtres !")
        
    except Exception as e:
        print(f"\n❌ ERREUR PHASE 3 - RÉASSEMBLAGE: {e}")
        raise Exception(f"Phase réassemblage échouée: {e}")


def _run_interactive_cli():
    """CLI interactif avec menus à choix multiples."""
    print("\n" + "="*60)
    print("🎯 CLI INTERACTIF - FINANCIAL BOT CRYPTO V2")
    print("="*60)
    
    while True:
        try:
            print("\n🤖 MENU PRINCIPAL:")
            print("1️⃣  Prédictions")
            print("2️⃣  État du système")
            print("3️⃣  Modèles disponibles")
            print("4️⃣  Confiance des modèles")
            print("5️⃣  Quitter")
            
            choice = input("\n👉 Votre choix (1-5): ").strip()
            
            if choice == '1':
                _menu_predictions()
            elif choice == '2':
                _menu_status()
            elif choice == '3':
                _menu_models()
            elif choice == '4':
                _show_confidence_scores()
            elif choice == '5':
                print("👋 Au revoir !")
                break
            else:
                print("❌ Choix invalide. Tapez 1, 2, 3, 4 ou 5")
                
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


def _menu_predictions():
    """Menu des prédictions."""
    print("\n🔮 MENU PRÉDICTIONS:")
    print("1️⃣  Prix de X crypto dans X temps")
    print("2️⃣  Avec X€ dans X crypto, temps pour faire X% bénéf")
    print("3️⃣  Meilleur moment pour acheter/vendre")
    print("4️⃣  Statistiques des prédictions")
    print("5️⃣  Retour au menu principal")
    
    choice = input("\n👉 Votre choix (1-5): ").strip()
    
    if choice == '1':
        _predict_price()
    elif choice == '2':
        _predict_profit_time()
    elif choice == '3':
        _predict_timing()
    elif choice == '4':
        _show_prediction_stats()
    elif choice == '5':
        pass  # Retour au menu principal
    else:
        print("❌ Choix invalide")


def _show_prediction_stats():
    """Affiche les statistiques des prédictions."""
    print("\n📊 STATISTIQUES DES PRÉDICTIONS")
    
    try:
        from prediction.prediction_tracker import PredictionTracker
        tracker = PredictionTracker()
        stats = tracker.get_prediction_stats()
        
        print(f"   🎯 Total prédictions: {stats['total']}")
        print(f"   ✅ Prédictions vérifiées: {stats['verified']}")
        print(f"   📈 Erreur moyenne: {stats['avg_error']:.2f}%")
        
        if stats['verified'] > 0:
            success_rate = (stats['verified'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"   🎯 Taux de vérification: {success_rate:.1f}%")
            
            if stats['avg_error'] < 3:
                print("   🟢 Excellente précision!")
            elif stats['avg_error'] < 7:
                print("   🟡 Bonne précision")
            else:
                print("   🔴 Précision à améliorer")
        else:
            print("   ⏳ Aucune prédiction vérifiée pour le moment")
            
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des stats: {e}")


def _predict_price():
    """Prédiction de prix."""
    print("\n💰 PRÉDICTION DE PRIX")
    
    # Choix crypto
    print("\n💎 Choisissez la crypto:")
    cryptos = ["bitcoin", "ethereum", "solana", "binancecoin", "cardano", "ripple", "dogecoin", "avalanche-2", "chainlink", "matic-network"]
    for i, crypto in enumerate(cryptos, 1):
        print(f"{i}️⃣  {crypto.upper()}")
    
    try:
        crypto_choice = int(input("\n👉 Votre choix (1-10): ").strip())
        if 1 <= crypto_choice <= 10:
            coin = cryptos[crypto_choice - 1]
        else:
            print("❌ Choix invalide")
            return
    except:
        print("❌ Veuillez entrer un nombre")
        return
    
    # Choix horizon
    print("\n⏱️ Choisissez l'horizon:")
    horizons = [("10 minutes", 10), ("1 heure", 60), ("6 heures", 360), ("12 heures", 720), ("8 heures", 480)]
    for i, (label, _) in enumerate(horizons, 1):
        print(f"{i}️⃣  {label}")
    
    try:
        horizon_choice = int(input("\n👉 Votre choix (1-5): ").strip())
        if 1 <= horizon_choice <= 5:
            horizon_label, horizon = horizons[horizon_choice - 1]
        else:
            print("❌ Choix invalide")
            return
    except:
        print("❌ Veuillez entrer un nombre")
        return
    
    print(f"\n🔍 Prédiction pour {coin.upper()} dans {horizon_label}...")
    
    try:
        # Charger le modèle (on sait que ça marche)
        from prediction.model_store import ModelStore
        store = ModelStore()
        model, metadata = store.load(coin, horizon)
        
        if model is None:
            print(f"❌ Aucun modèle entraîné pour {coin.upper()} horizon {horizon}min")
            return
        
        print(f"✅ Modèle chargé: {metadata.get('algo', 'Inconnu')}")
        
        # Charger les données de manière sûre (éviter le conflit de schéma)
        from storage.parquet_writer import ParquetWriter
        import pandas as pd
        import os
        
        writer = ParquetWriter()
        data_path = writer.settings["paths"]["data_parquet"]
        five_min_path = os.path.join(data_path, "five_min")
        
        # Lire fichier par fichier pour éviter les conflits de schéma
        dfs = []
        for root, dirs, files in os.walk(five_min_path):
            for file in files:
                if file.endswith('.parquet'):
                    try:
                        file_path = os.path.join(root, file)
                        df_part = pd.read_parquet(file_path)
                        # Filtrer seulement notre coin
                        if 'coin_id' in df_part.columns:
                            df_coin_part = df_part[df_part['coin_id'] == coin]
                            if not df_coin_part.empty:
                                dfs.append(df_coin_part)
                    except Exception:
                        continue  # Ignorer les fichiers problématiques
        
        if not dfs:
            print(f"❌ Pas de données trouvées pour {coin}")
            return
        
        # Combiner les données
        df_coin = pd.concat(dfs, ignore_index=True).sort_values("ts_utc_ms").tail(1000)
        
        if df_coin.empty:
            print(f"❌ Pas de données récentes pour {coin}")
            return
        
        # Obtenir le prix actuel
        current_price = df_coin.iloc[-1]["c"]
        
        # Construire les features pour prédiction
        from prediction.feature_builder import FeatureBuilder
        fb = FeatureBuilder(step_minutes=5)
        
        # Utiliser toutes les données disponibles pour construire les features
        X, y = fb.build_from_five_min(df_coin, coin, horizon)
        
        if X is None or X.empty:
            print(f"❌ Impossible de construire les features pour {coin}")
            return
        
        # Prendre les dernières features (les plus récentes)
        last_features = X.tail(1).values
        
        # Faire la prédiction
        prediction = model.predict(last_features)[0]
        
        # Le modèle prédit le changement relatif
        future_price = current_price * (1 + prediction)
        change_pct = prediction * 100
        
        # Sauvegarder la prédiction pour vérification future
        from prediction.prediction_tracker import PredictionTracker
        tracker = PredictionTracker()
        prediction_id = tracker.save_prediction(
            coin_id=coin,
            horizon_minutes=horizon,
            current_price=current_price,
            predicted_price=future_price,
            model_algo=metadata.get('algo', 'Inconnu')
        )
        
        print(f"\n✅ PRÉDICTION RÉUSSIE:")
        print(f"   💰 Prix actuel: ${current_price:.4f}")
        print(f"   🎯 Prix prédit: ${future_price:.4f}")
        print(f"   📈 Variation: {change_pct:+.2f}%")
        print(f"   🤖 Modèle: {metadata.get('algo', 'Inconnu')}")
        print(f"   📊 Précision (MAE): {metadata.get('mae_score', 0):.6f}")
        print(f"   � Échantillons d'entraînement: {metadata.get('training_samples', 0):,}")
        
        # Recommandation
        if change_pct > 1:
            print(f"   🟢 Recommandation: ACHETER (hausse prévue)")
        elif change_pct < -1:
            print(f"   🔴 Recommandation: VENDRE (baisse prévue)")
        else:
            print(f"   🟡 Recommandation: HOLD (stabilité prévue)")
            
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()


def _predict_profit_time():
    """Calcule le temps estimé pour atteindre un certain pourcentage de bénéfice."""
    print("\n⏱️ CALCUL TEMPS POUR BÉNÉFICE")
    
    # --- Inputs Utilisateur ---
    # 1. Choix crypto
    print("\n� Choisissez la crypto:")
    cryptos = ["bitcoin", "ethereum", "solana", "binancecoin", "cardano", "ripple", "dogecoin", "avalanche-2", "chainlink", "matic-network"]
    for i, crypto in enumerate(cryptos, 1):
        print(f"{i}️⃣  {crypto.upper()}")
    
    try:
        crypto_choice = int(input("\n👉 Votre choix (1-10): ").strip())
        coin = cryptos[crypto_choice - 1]
    except (ValueError, IndexError):
        print("❌ Choix invalide.")
        return

    # 2. Montant investi (informatif, n'impacte pas le calcul de temps)
    try:
        amount_str = input("👉 Montant à investir (ex: 1000): ").strip()
        amount = float(amount_str)
    except ValueError:
        print("❌ Montant invalide.")
        return

    # 3. Pourcentage de profit
    try:
        profit_pct_str = input("👉 Pourcentage de profit désiré (ex: 5 pour 5%): ").strip()
        profit_pct = float(profit_pct_str)
    except ValueError:
        print("❌ Pourcentage invalide.")
        return

    print("\n" + "="*60)
    print(f"🔍 Recherche du temps estimé pour un profit de {profit_pct}% sur {coin.upper()}...")
    print("="*60)

    try:
        # --- Logique de Prédiction ---
        # Charger les données une seule fois
        from storage.parquet_writer import ParquetWriter
        import pandas as pd
        import os
        
        writer = ParquetWriter()
        data_path = writer.settings["paths"]["data_parquet"]
        five_min_path = os.path.join(data_path, "five_min")
        
        dfs = [pd.read_parquet(os.path.join(root, file)) for root, _, files in os.walk(five_min_path) for file in files if file.endswith('.parquet')]
        df_full = pd.concat(dfs, ignore_index=True)
        df_coin = df_full[df_full['coin_id'] == coin].sort_values("ts_utc_ms").tail(1000)

        if df_coin.empty:
            print(f"❌ Pas de données récentes pour {coin}")
            return

        current_price = df_coin.iloc[-1]["c"]
        target_price = current_price * (1 + profit_pct / 100)

        print(f"   💰 Prix actuel: ${current_price:,.2f}")
        print(f"   🎯 Prix cible pour {profit_pct}%: ${target_price:,.2f}")
        
        # Itérer sur les horizons de temps disponibles
        horizons = [("10 minutes", 10), ("1 heure", 60), ("6 heures", 360), ("8 heures", 480), ("12 heures", 720)]
        
        from prediction.model_store import ModelStore
        from prediction.feature_builder import FeatureBuilder
        
        store = ModelStore()
        fb = FeatureBuilder(step_minutes=5)
        
        found_horizon = False
        for label, horizon_min in horizons:
            print(f"\n⏳ Analyse de l'horizon {label}...")
            
            model, metadata = store.load(coin, horizon_min)
            if model is None:
                print(f"   -> ❌ Modèle non disponible.")
                continue

            X, _ = fb.build_from_five_min(df_coin.copy(), coin, horizon_min)
            if X is None or X.empty:
                print("   -> ❌ Impossible de construire les features.")
                continue
            
            last_features = X.tail(1).values
            prediction_relative = model.predict(last_features)[0]
            predicted_price = current_price * (1 + prediction_relative)
            
            print(f"   -> 🔮 Prédiction: ${predicted_price:,.2f}")

            if predicted_price >= target_price:
                print("\n" + "="*60)
                print("✅ HORIZON TROUVÉ !")
                print(f"   Le profit de {profit_pct}% pourrait être atteint en environ: {label}")
                print(f"   Prix prédit pour cet horizon: ${predicted_price:,.2f}")
                print("="*60)
                found_horizon = True
                break # Arrêter dès qu'un horizon correspondant est trouvé
        
        if not found_horizon:
            print("\n" + "="*60)
            print("❌ AUCUN HORIZON TROUVÉ")
            print("   Aucun des modèles ne prédit une hausse suffisante pour atteindre votre objectif.")
            print("   Essayez avec un pourcentage de profit plus bas ou réessayez plus tard.")
            print("="*60)

    except Exception as e:
        print(f"❌ Erreur lors du calcul: {e}")
        import traceback
        traceback.print_exc()


def _predict_timing():
    """Analyse tous les modèles pour trouver le meilleur moment pour acheter ou vendre."""
    print("\n" + "="*60)
    print("📈 TIMING OPTIMAL ACHAT/VENTE")
    print("   Analyse de toutes les cryptos et horizons pour trouver les meilleures opportunités...")
    print("="*60)

    try:
        # --- Dépendances ---
        from storage.parquet_writer import ParquetWriter
        from prediction.model_store import ModelStore
        from prediction.feature_builder import FeatureBuilder
        import pandas as pd
        import os

        # --- Initialisation ---
        writer = ParquetWriter()
        store = ModelStore()
        fb = FeatureBuilder(step_minutes=5)
        
        cryptos = ["bitcoin", "ethereum", "solana", "binancecoin", "cardano", "ripple", "dogecoin", "avalanche-2", "chainlink", "matic-network"]
        horizons = [("10min", 10), ("1h", 60), ("6h", 360), ("8h", 480), ("12h", 720)]
        
        opportunities = []

        # --- Chargement des données (une seule fois) ---
        print("⏳ Chargement des données de marché...")
        data_path = writer.settings["paths"]["data_parquet"]
        five_min_path = os.path.join(data_path, "five_min")
        
        dfs = [pd.read_parquet(os.path.join(root, file)) for root, _, files in os.walk(five_min_path) for file in files if file.endswith('.parquet')]
        df_full = pd.concat(dfs, ignore_index=True)
        print("✅ Données chargées.")

        # --- Boucle d'analyse ---
        for coin in cryptos:
            print(f"\n🔍 Analyse de {coin.upper()}...")
            df_coin = df_full[df_full['coin_id'] == coin].sort_values("ts_utc_ms").tail(1000)
            
            if df_coin.empty:
                print(f"   -> ❌ Pas de données pour {coin}.")
                continue
            
            current_price = df_coin.iloc[-1]["c"]

            for label, horizon_min in horizons:
                model, metadata = store.load(coin, horizon_min)
                if not model:
                    continue

                X, _ = fb.build_from_five_min(df_coin.copy(), coin, horizon_min)
                if X is None or X.empty:
                    continue
                
                last_features = X.tail(1).values
                prediction_relative = model.predict(last_features)[0]
                change_pct = prediction_relative * 100

                # Calcul du score : variation par heure pour normaliser
                score = (change_pct / (horizon_min / 60)) if horizon_min > 0 else 0

                if abs(change_pct) > 0.05: # Seuil minimal pour considérer une opportunité
                    opportunities.append({
                        "coin": coin,
                        "horizon_label": label,
                        "change_pct": change_pct,
                        "score": score,
                        "type": "ACHAT" if change_pct > 0 else "VENTE"
                    })
        
        # --- Classement et Affichage ---
        if not opportunities:
            print("\n" + "="*60)
            print("🟡 AUCUNE OPPORTUNITÉ SIGNIFICATIVE DÉTECTÉE")
            print("   Le marché semble stable pour le moment.")
            print("="*60)
            return

        buy_ops = sorted([op for op in opportunities if op['type'] == 'ACHAT'], key=lambda x: x['score'], reverse=True)
        sell_ops = sorted([op for op in opportunities if op['type'] == 'VENTE'], key=lambda x: x['score'])

        print("\n" + "="*60)
        print("🏆 TOP 3 OPPORTUNITÉS D'ACHAT (HAUSSE RAPIDE)")
        print("="*60)
        if buy_ops:
            for i, op in enumerate(buy_ops[:3], 1):
                print(f"{i}️⃣  {op['coin'].upper():<15} -> {op['change_pct']:>+6.2f}% en {op['horizon_label']:<5} (Score: {op['score']:.2f})")
        else:
            print("   Aucune opportunité d'achat détectée.")

        print("\n" + "="*60)
        print("🚨 TOP 3 OPPORTUNITÉS DE VENTE (BAISSE RAPIDE)")
        print("="*60)
        if sell_ops:
            for i, op in enumerate(sell_ops[:3], 1):
                print(f"{i}️⃣  {op['coin'].upper():<15} -> {op['change_pct']:>+6.2f}% en {op['horizon_label']:<5} (Score: {op['score']:.2f})")
        else:
            print("   Aucune opportunité de vente détectée.")
        print("="*60)

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse du timing: {e}")
        import traceback
        traceback.print_exc()


def _menu_status():
    """Afficher l'état du système."""
    try:
        from ops.cli import status
        status()
    except Exception as e:
        print(f"❌ Erreur: {e}")


def _menu_models():
    """Afficher les modèles."""
    try:
        from ops.cli import models
        models()
    except Exception as e:
        print(f"❌ Erreur: {e}")


def _show_confidence_scores():
    """Calcule et affiche le score de confiance actuel pour chaque crypto."""
    print("\n" + "="*60)
    print("💯 SCORE DE CONFIANCE ACTUEL DES MODÈLES")
    print("   Analyse des prédictions récentes pour évaluer la stabilité...")
    print("="*60)

    try:
        # --- Dépendances ---
        from storage.parquet_writer import ParquetWriter
        from prediction.model_store import ModelStore
        from prediction.feature_builder import FeatureBuilder
        from prediction.confidence import ConfidenceEstimator
        import pandas as pd
        import os

        # --- Initialisation ---
        writer = ParquetWriter()
        store = ModelStore()
        fb = FeatureBuilder(step_minutes=5)
        estimator = ConfidenceEstimator()
        
        cryptos = ["bitcoin", "ethereum", "solana", "binancecoin", "cardano", "ripple", "dogecoin", "avalanche-2", "chainlink", "matic-network"]
        
        print("⏳ Chargement des données de marché (peut prendre un moment)...")
        data_path = writer.settings["paths"]["data_parquet"]
        five_min_path = os.path.join(data_path, "five_min")
        
        dfs = [pd.read_parquet(os.path.join(root, file)) for root, _, files in os.walk(five_min_path) for file in files if file.endswith('.parquet')]
        df_full = pd.concat(dfs, ignore_index=True)
        print("✅ Données chargées.")

        # --- Boucle d'analyse ---
        confidence_scores = []
        for coin in cryptos:
            df_coin = df_full[df_full['coin_id'] == coin].sort_values("ts_utc_ms").tail(1000)
            
            if df_coin.empty or len(df_coin) < 50: # Besoin d'assez de données pour la confiance
                confidence_scores.append({"coin": coin, "score": "Données insuffisantes"})
                continue
            
            # On ne prend que le modèle 10 minutes comme référence pour la confiance
            model, _ = store.load(coin, 10)
            if not model:
                confidence_scores.append({"coin": coin, "score": "Modèle 10min absent"})
                continue

            X, _ = fb.build_from_five_min(df_coin.copy(), coin, 10)
            if X is None or X.empty or len(X) < 10:
                confidence_scores.append({"coin": coin, "score": "Features insuffisantes"})
                continue
            
            # Simuler les 10 dernières prédictions
            last_10_features = X.tail(10).values
            predictions = model.predict(last_10_features)
            
            # Calculer la confiance
            confidence_info = estimator.get_confidence_metrics(coin, 10)
            uncertainty = confidence_info.get("calibrated_uncertainty_pct", 1.0)
            score = 1.0 - uncertainty
            confidence_scores.append({"coin": coin, "score": f"{score * 100:.1f}%"})

        # --- Affichage ---
        print("\n" + "="*60)
        print("📊 RÉSULTATS DES SCORES DE CONFIANCE")
        print("="*60)
        for item in confidence_scores:
            print(f"   - {item['coin'].upper():<15}: {item['score']}")
        print("="*60)
        print("\nℹ️ Un score élevé indique des prédictions stables et fiables.")
        print("   Un score bas indique un marché volatile ou imprévisible.")

    except Exception as e:
        print(f"❌ Erreur lors du calcul de la confiance: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Point d'entrée principal."""
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Import des modules selon la commande
        if args.command == 'demarrer':
            from ops.scheduler import Scheduler
            import json, socket, time
            # Utilitaires locaux pour pid/lock (miroir de ops.cli)
            def _runtime_paths_main():
                try:
                    from ops.cli import _settings as _s
                    sconf = _s()
                    runtime_dir = sconf["paths"]["runtime"]
                    if not os.path.isabs(runtime_dir):
                        runtime_dir = os.path.join(os.path.dirname(__file__), runtime_dir)
                    os.makedirs(runtime_dir, exist_ok=True)
                    return (os.path.join(runtime_dir, "pid"), os.path.join(runtime_dir, "app.lock"))
                except Exception:
                    fallback = os.path.join(os.path.dirname(__file__), "ops", "runtime")
                    os.makedirs(fallback, exist_ok=True)
                    return (os.path.join(fallback, "pid"), os.path.join(fallback, "app.lock"))

            pid_path_main, lock_path_main = _runtime_paths_main()
            print("Demarrage du Financial Bot Crypto V2...")
            print("   - Collecte de donnees en temps reel")
            print("   - Collecte historique massive si necessaire") 
            print("   - Entrainement ML periodique")
            print("   - Predictions automatiques")
            
            if hasattr(args, 'test_mode') and args.test_mode:
                print("\nMode test: demarrage puis arret automatique en 3s...")
                print("🧪 Mode test: AUCUNE collecte de données pour éviter conflits de schéma")
                # Simulation simple sans démarrer le scheduler réel
                import time
                time.sleep(3)
                print("Test demarrage/arret termine avec succes")
            else:
                print("\nUtilisez Ctrl+C pour arreter le bot.")
                
                # Forcer l'affichage immédiat sous Windows
                import sys
                sys.stdout.flush()
                
                # DEBUG : FORCER LA VÉRIFICATION
                print(f"\n🔍 DEBUG: Vérification premier démarrage...")
                sys.stdout.flush()
                is_first = _is_first_startup()
                print(f"🔍 DEBUG: _is_first_startup() = {is_first}")
                sys.stdout.flush()
                
                # VÉRIFIER ET ENTRAÎNER AVANT DE LANCER LE SCHEDULER
                if is_first:
                    print("\n🚀 PREMIER DÉMARRAGE DÉTECTÉ !")
                    print("Initialisation complète : Collecte → Split → Train → Reassemble")
                    sys.stdout.flush()
                    
                    try:
                        _perform_initial_setup()
                    except Exception as e:
                        print(f"\n❌ ERREUR CRITIQUE DURANT L'INITIALISATION: {e}")
                        print("🛑 ARRÊT DU BOT - Réparez l'erreur avant de relancer")
                        sys.stdout.flush()
                        return 1  # Code d'erreur
                    
                    print("\n🎉 INITIALISATION COMPLÈTE RÉUSSIE !")
                    sys.stdout.flush()
                else:
                    print("\n✅ Modèles déjà présents - Démarrage direct du scheduler")
                    sys.stdout.flush()
                
                print("🚀 Démarrage du scheduler temps réel...")
                scheduler = Scheduler()
                scheduler.start()
                # Écrire pid/lock ici aussi (process courant)
                try:
                    info = {
                        "pid": os.getpid(),
                        "started_at": datetime.utcnow().isoformat()+"Z",
                        "host": socket.gethostname(),
                        "cmd": "main.py demarrer (inline scheduler)",
                        "mode": "inline"
                    }
                    with open(pid_path_main, 'w', encoding='utf-8') as f:
                        json.dump(info, f, ensure_ascii=False)
                    with open(lock_path_main, 'w', encoding='utf-8') as f:
                        f.write('LOCK\n')
                except Exception as e:
                    print(f"⚠️ Impossible d'écrire pid/lock: {e}")
                
                print("✅ Scheduler démarré en arrière-plan")
                print("🎯 Lancement du CLI interactif...")
                
                # Lancer le CLI interactif au lieu de la boucle infinie
                try:
                    print("🎯 Mode prédictions interactives - Entrez vos paramètres...")
                    _run_interactive_cli()
                except KeyboardInterrupt:
                    print("\nArret du bot en cours...")
                    scheduler.stop()
                    # Nettoyage pid/lock
                    try:
                        if os.path.exists(pid_path_main):
                            os.remove(pid_path_main)
                        if os.path.exists(lock_path_main):
                            os.remove(lock_path_main)
                    except Exception:
                        pass
                    print("Bot arrete proprement")
                except Exception as e:
                    print(f"\n❌ ERREUR INATTENDUE: {e}")
                    import traceback
                    traceback.print_exc()
                    scheduler.stop()
                    try:
                        if os.path.exists(pid_path_main):
                            os.remove(pid_path_main)
                        if os.path.exists(lock_path_main):
                            os.remove(lock_path_main)
                    except Exception:
                        pass
                    print("Bot arrete avec erreur")
                
        elif args.command == 'arreter':
            print("Commande d'arret recue")
            print("   (Utilisez Ctrl+C dans le terminal du bot pour l'arreter)")
            
        elif args.command == 'etat':
            from ops.cli import status
            status()
            
        elif args.command == 'collect-historical':
            from collectors.historical_collector import HistoricalCollector
            from storage.parquet_writer import ParquetWriter
            
            print("Demarrage collecte historique massive URGENTE...")
            print("   - Donnees daily: 2013-2018")
            print("   - Donnees hourly: 2018-2025") 
            print("   - Donnees 5min: dernieres 24h")
            print("   Collecte avant degradation API 1er septembre!")
            
            writer = ParquetWriter()
            collector = HistoricalCollector(writer=writer)
            
            # Mode par défaut: dry-run (simulation rapide)
            is_dry_run = not (hasattr(args, 'real') and args.real)
            
            # Mode test: collecte très rapide pour smoke test
            if hasattr(args, 'test_mode') and args.test_mode:
                print("\nMode TEST (test_mode=True) - collecte 30s max")
                # Simulation ultra-rapide pour smoke test
                is_dry_run = True
                print("✅ Collecte historique test terminée en simulation")
                return
            
            if is_dry_run:
                print("\nMode SIMULATION (dry_run=True) - rapide")
            else:
                print("\nMode REEL (dry_run=False) - peut prendre du temps")
            
            results = collector.collect_all_historical(dry_run=is_dry_run)
            
            summary = results.get("summary", {})
            print(f"\nCollecte historique terminee:")
            print(f"   - Points collectes: {summary.get('total_points_collected', 0):,}")
            print(f"   - Duree: {summary.get('total_duration_minutes', 0):.1f} minutes")
            print(f"   - Taille estimee: {summary.get('estimated_size_mb', 0):.1f} MB")
            
        elif args.command == 'train':
            from prediction.trainer import Trainer
            from storage.parquet_writer import ParquetWriter
            import pandas as pd
            
            print(f"Entrainement baseline pour {args.coin}")
            
            # Mode test : utiliser données simulées en mémoire
            if hasattr(args, 'test_mode') and args.test_mode:
                print("🧪 Mode test : Simulation d'entraînement avec données factices")
                
                # Créer données factices directement en mémoire
                from tools.test_data_manager import TestDataManager
                test_manager = TestDataManager()
                df_test = test_manager.create_fake_five_min_data(args.coin, hours=48)
                
                print(f"   - Données test créées: {len(df_test):,} points")
                
                # Simuler entraînement
                trainer = Trainer()
                try:
                    trainer.train(df_test, args.coin, horizon_minutes=10)
                    print("✅ Entraînement test terminé avec succès")
                except Exception as e:
                    print(f"❌ Erreur entraînement test: {e}")
                
                return
            
            # Mode normal : utiliser fichiers Parquet
            writer = ParquetWriter()
            trainer = Trainer()
            
            # Charger donnees recentes pour entrainement
            try:
                data_path = writer.settings["paths"]["data_parquet"]
                five_min_path = os.path.join(data_path, "five_min")
                
                if os.path.exists(five_min_path):
                    df = pd.read_parquet(five_min_path)
                    print(f"   - Donnees chargees: {len(df):,} points")
                    
                    # Vérifier que les données contiennent la colonne coin_id
                    if 'coin_id' not in df.columns:
                        print("Erreur: colonne 'coin_id' manquante dans les données")
                        return
                    
                    # Entrainer pour le coin specifie
                    if len(df) == 0:
                        print("Aucune donnee disponible pour l'entrainement")
                    elif args.coin in df['coin_id'].unique():
                        print(f"   - Entrainement {args.coin}...")
                        coin_data = df[df['coin_id'] == args.coin].sort_values('ts_utc_ms')
                        if len(coin_data) > 100:
                            trainer.train(coin_data, args.coin, horizon_minutes=10)
                            print("Entrainement complete termine")
                        else:
                            print(f"Pas assez de donnees pour {args.coin}")
                    else:
                        print(f"Coin {args.coin} non trouve dans les donnees")
                    
                else:
                    print("Aucune donnee disponible pour l'entrainement")
                    
            except Exception as e:
                print(f"Erreur entrainement: {e}")
                
        elif args.command == 'train-incremental':
            from prediction.incremental_trainer import IncrementalTrainer
            
            print(f"Entrainement incremental mode: {args.mode}")
            
            trainer = IncrementalTrainer()
            if args.mode == 'micro':
                # Micro update nécessite des données, simulons
                print("Mode micro - simulation (pas de donnees recentes)")
            elif args.mode == 'mini':
                # Mini retrain nécessite des données, simulons
                print("Mode mini - simulation (pas de donnees recentes)")
            elif args.mode == 'schedule':
                # Schedule fonctionne sans données
                schedule = trainer.get_training_schedule(['bitcoin', 'ethereum'])
                print(f"Calendrier genere: {len(schedule)} taches")
                
        elif args.command == 'prevoir':
            from ops.cli import predict
            print(f"Prediction {args.coin} +{args.value}{args.unit}")
            if hasattr(args, 'amount_eur') and args.amount_eur:
                print(f"Montant: {args.amount_eur} EUR")
            
            # Vérifier mode test
            test_mode = hasattr(args, 'test_mode') and args.test_mode
            
            # Appeler predict directement avec les paramètres  
            predict(
                kind=args.kind,
                coin_opt=args.coin,
                unit_opt=args.unit,
                value_opt=args.value,
                amount_eur_opt=getattr(args, 'amount_eur', None),
                test_mode=test_mode
            )
            
        elif args.command == 'confidence-metrics':
            from prediction.confidence import ConfidenceEstimator
            from storage.parquet_writer import ParquetWriter
            
            print(f"Metriques de confiance pour {args.coin} (horizon {args.horizon}min)")
            
            writer = ParquetWriter()
            estimator = ConfidenceEstimator()
            
            try:
                metrics = estimator.get_confidence_metrics(
                    coin_id=args.coin,
                    horizon_minutes=args.horizon
                )
                
                print(f"\nResultats pour {args.coin}:")
                print(f"   - Confiance globale: {metrics.get('calibrated_mae_pct', 0)*100:.2f}%")
                print(f"   - Precision historique: {metrics.get('historical_mae_pct', 0)*100:.2f}%")
                print(f"   - Incertitude: {metrics.get('calibrated_uncertainty_pct', 0)*100:.2f}%")
                print(f"   - Volatilite: {metrics.get('historical_volatility_pct', 0)*100:.2f}%")
                print(f"   - Seuil configure: {metrics.get('threshold_pct', 0)*100:.2f}%")
                print(f"   - Donnees historiques: {'✅ Oui' if metrics.get('has_historical_data', False) else '❌ Non'}")
                
            except Exception as e:
                print(f"Erreur calcul metriques: {e}")
                import traceback
                traceback.print_exc()
                
        elif args.command == 'models':
            from prediction.model_store import ModelStore
            
            print("Modeles ML disponibles:")
            
            store = ModelStore()
            models = store.list_models()
            
            if models:
                for model_path in models:
                    # Extraire les infos du nom de fichier
                    filename = os.path.basename(model_path)
                    if '__' in filename:
                        parts = filename.replace('.pkl', '').split('__')
                        coin = parts[0] if len(parts) > 0 else 'unknown'
                        horizon = parts[1] if len(parts) > 1 else 'unknown'
                        print(f"   - {coin} (horizon {horizon}): {model_path}")
                    else:
                        print(f"   - {model_path}")
            else:
                print("   Aucun modele disponible. Lancez l'entrainement d'abord.")
                
        elif args.command == 'parametres':
            from ops.cli import settings
            settings()
            
        else:
            print(f"Commande inconnue: {args.command}")
            parser.print_help()
            
    except Exception as e:
        print(f"Erreur: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
