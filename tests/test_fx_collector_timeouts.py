"""
Tests pour le FxCollector assurant les timeouts complets sur la configuration httpx client
"""

from src.collectors.fx_collector import FxCollector


def test_fx_collector_client_uses_complete_timeout_config():
    """Test que le client httpx utilise bien tous les timeouts requis"""
    collector = FxCollector()
    client = collector._client()
    
    # Vérifie que l'objet timeout contient les 4 valeurs requises par httpx
    assert client.timeout.connect is not None, "connect timeout devrait être configuré"
    assert client.timeout.read is not None, "read timeout devrait être configuré"
    assert client.timeout.write is not None, "write timeout devrait être configuré"
    assert client.timeout.pool is not None, "pool timeout devrait être configuré"
    
    # Les valeurs devraient être conformes à ce qui est spécifié dans api_config.json
    # ou prendre des valeurs par défaut raisonnables
    assert client.timeout.connect > 0, "connect timeout devrait être positif"
    assert client.timeout.read > 0, "read timeout devrait être positif"
    assert client.timeout.write > 0, "write timeout devrait être positif"
    assert client.timeout.pool > 0, "pool timeout devrait être positif"
