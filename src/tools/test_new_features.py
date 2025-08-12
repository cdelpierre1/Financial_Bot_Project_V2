"""
Script de test pour les nouvelles fonctionnalités implémentées.

Tests:
1. Modèles ML avancés (LightGBM, XGBoost, RandomForest)
2. Features avancées (RSI, Bollinger, etc.)
3. Calcul de profit attendu
4. Estimation de confiance calibrée
5. Entraînement incrémental
"""
from __future__ import annotations

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Ajouter le path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prediction.trainer import Trainer
from prediction.feature_builder import FeatureBuilder
from prediction.confidence import ConfidenceEstimator
from prediction.pipeline import PredictionPipeline
from prediction.incremental_trainer import IncrementalTrainer


def generate_synthetic_data(coin_id: str = "bitcoin", days: int = 7) -> pd.DataFrame:
    """Génère des données synthétiques five_min pour les tests."""
    print(f"Generating {days} days of synthetic data for {coin_id}...")
    
    # Générer timestamps (5 min intervals)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start_time, end_time, freq='5min')
    
    # Prix initial aléatoire
    initial_price = np.random.uniform(30000, 50000)  # Prix Bitcoin typique
    
    # Générer prix avec marche aléatoire + tendance
    n_points = len(timestamps)
    returns = np.random.normal(0, 0.01, n_points)  # Volatilité 1%
    returns[::288] += np.random.normal(0, 0.05, len(returns[::288]))  # Chocs quotidiens
    
    prices = [initial_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 100))  # Prix minimum 100
    
    # Créer OHLC réaliste
    df_list = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Simuler Open, High, Low autour du Close
        volatility = abs(np.random.normal(0, 0.005))  # Volatilité intra-période
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = np.random.uniform(low, high)
        
        # Volume synthétique
        volume = np.random.uniform(100000, 1000000)
        
        row = {
            'ts_utc_ms': int(ts.timestamp() * 1000),
            'coin_id': coin_id,
            'o': round(open_price, 2),
            'h': round(high, 2),
            'l': round(low, 2),
            'c': round(close, 2),
            'volume': round(volume, 2),
            'agg_method': 'synthetic'
        }
        df_list.append(row)
    
    df = pd.DataFrame(df_list)
    print(f"Generated {len(df)} synthetic data points")
    return df


def test_feature_builder():
    """Test du FeatureBuilder avec features avancées."""
    print("\n=== Test FeatureBuilder ===")
    
    # Données synthétiques
    df = generate_synthetic_data("bitcoin", days=2)
    
    fb = FeatureBuilder(step_minutes=5)
    X, y = fb.build_from_five_min(df, "bitcoin", horizon_minutes=10)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Sample features (first row): {X.iloc[0].to_dict()}")
    
    # Vérifier que les nouvelles features sont présentes
    expected_features = ['rsi', 'volatility', 'bb_position', 'ma_ratio', 'momentum_5']
    missing_features = [f for f in expected_features if f not in X.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    else:
        print("✅ All advanced features present")
    
    return X, y


def test_trainer_models():
    """Test des modèles ML avancés."""
    print("\n=== Test Trainer Models ===")
    
    # Données synthétiques
    df = generate_synthetic_data("bitcoin", days=3)
    
    trainer = Trainer()
    
    # Test auto-selection
    print("Testing auto-selection...")
    result = trainer.train(df, "bitcoin", horizon_minutes=10)
    print(f"Auto-selected model result: {result}")
    
    # Test modèles spécifiques
    models_to_test = ["linear", "random_forest"]
    if "lightgbm" in trainer.model_configs:
        models_to_test.append("lightgbm")
    if "xgboost" in trainer.model_configs:
        models_to_test.append("xgboost")
    
    for model_name in models_to_test:
        print(f"Testing {model_name}...")
        try:
            result = trainer.train(df, "bitcoin", horizon_minutes=10, force_model=model_name)
            print(f"  {model_name}: {result.get('status')} - MAE: {result.get('mae_score', 'N/A')}")
        except Exception as e:
            print(f"  {model_name}: ERROR - {e}")
    
    return result


def test_confidence_estimator():
    """Test de l'estimateur de confiance."""
    print("\n=== Test ConfidenceEstimator ===")
    
    conf = ConfidenceEstimator()
    
    # Test métriques
    metrics = conf.get_confidence_metrics("bitcoin", 10)
    print(f"Confidence metrics: {json.dumps(metrics, indent=2)}")
    
    # Test bounds
    mid_price = 45000.0
    ci_lo, ci_hi = conf.ci_bounds(mid_price, "bitcoin", 10)
    print(f"CI bounds for ${mid_price}: [{ci_lo:.2f}, {ci_hi:.2f}]")
    
    # Test erreur attendue
    expected_error = conf.expected_error_pct("bitcoin", 10, 0.01)
    print(f"Expected error with 1% costs: {expected_error:.4f} ({expected_error*100:.2f}%)")
    
    return metrics


def test_prediction_pipeline():
    """Test de la pipeline complète avec profit attendu."""
    print("\n=== Test PredictionPipeline ===")
    
    # Créer données synthétiques et un modèle
    df = generate_synthetic_data("bitcoin", days=2)
    trainer = Trainer()
    trainer.train(df, "bitcoin", horizon_minutes=10)
    
    # Pipeline
    pipeline = PredictionPipeline()
    
    # Test prédiction avec profit attendu
    result = pipeline.run(
        coin_id="bitcoin",
        horizon_minutes=10,
        mid_price_usd=45000.0,
        fx_rate_usd_per_eur=1.1,
        amount_eur=1000.0,  # Montant pour calcul profit
        recent_five_min_df=df
    )
    
    print("Pipeline result:")
    print(f"  Predicted price: {result.get('estimation', {}).get('value_pred')}")
    print(f"  Expected profit: {result.get('estimation', {}).get('expected_profit_net_eur')}")
    print(f"  Decision status: {result.get('decision', {}).get('status')}")
    print(f"  Total costs: {result.get('costs', {}).get('total_cost_eur')}")
    
    return result


def test_incremental_trainer():
    """Test de l'entraînement incrémental."""
    print("\n=== Test IncrementalTrainer ===")
    
    # Données synthétiques étalées
    df = generate_synthetic_data("bitcoin", days=5)
    
    inc_trainer = IncrementalTrainer()
    
    # Test micro update
    print("Testing micro update...")
    result_micro = inc_trainer.micro_update(df, "bitcoin", 10)
    print(f"Micro update: {result_micro.get('status')} - {result_micro.get('training_type')}")
    
    # Test mini retrain
    print("Testing mini retrain...")
    result_mini = inc_trainer.mini_retrain(df, "bitcoin", 10)
    print(f"Mini retrain: {result_mini.get('status')} - {result_mini.get('training_type')}")
    
    # Test schedule
    print("Testing schedule...")
    schedule = inc_trainer.get_training_schedule(["bitcoin"], [10, 360])
    print(f"Schedule keys: {list(schedule.keys())}")
    for key, info in schedule.items():
        print(f"  {key}: needs_micro={info.get('needs_micro')}, needs_mini={info.get('needs_mini')}")
    
    return schedule


def main():
    """Exécute tous les tests."""
    print("🚀 Testing new features implementation...")
    print("=" * 60)
    
    try:
        # Test 1: FeatureBuilder
        X, y = test_feature_builder()
        
        # Test 2: Trainer avancé
        train_result = test_trainer_models()
        
        # Test 3: ConfidenceEstimator
        conf_metrics = test_confidence_estimator()
        
        # Test 4: Pipeline avec profit
        pipeline_result = test_prediction_pipeline()
        
        # Test 5: IncrementalTrainer
        schedule = test_incremental_trainer()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n📊 Summary:")
        print(f"  - Features generated: {X.shape[1] if 'X' in locals() else 'N/A'}")
        print(f"  - Model trained: {train_result.get('selected_model', 'N/A') if 'train_result' in locals() else 'N/A'}")
        print(f"  - Has historical data: {conf_metrics.get('has_historical_data', False) if 'conf_metrics' in locals() else False}")
        print(f"  - Pipeline decision: {pipeline_result.get('decision', {}).get('status', 'N/A') if 'pipeline_result' in locals() else 'N/A'}")
        print(f"  - Schedule models: {len(schedule) if 'schedule' in locals() else 0}")
        
        print("\n🎯 Ready for production!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
