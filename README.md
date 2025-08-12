# 🤖 Financial Bot Crypto V2 - Guide pour Débutants  
[![Projet d'apprentissage – Pas d'usage pro ni commercial](https://img.shields.io/badge/Projet%20d'apprentissage-Pas%20d'usage%20pro%20ni%20commercial-red)](#-sécurité--avertissements)  
[![Avertissement – Aucun conseil financier](https://img.shields.io/badge/Avertissement-Aucun%20conseil%20financier-orange)](#-sécurité--avertissements)  
[![Licence MIT Modifiée](https://img.shields.io/badge/Licence-MIT%20Modifi%C3%A9e-blue)](LICENSE)


## 📌 Description

**"🤖 Financial Bot Crypto V2 est un projet open source d'apprentissage combinant IA et cryptomonnaies. Il analyse et prédit les tendances pour explorer le marché. Aucun but commercial. N'investissez qu'après vos propres vérifications et seulement ce que vous pouvez vous permettre de perdre."**

---

## 📖 Présentation

Le **Financial Bot Crypto V2** est un assistant automatisé qui :  
- 📊 **Surveille** les prix des cryptomonnaies 24h/24  
- 🧠 **Apprend** des patterns du marché grâce à l'IA  
- 🎯 **Prédit** les mouvements futurs  
- 💡 **Conseille** les moments d'achat ou de vente  

Il s'agit d'un **projet d'apprentissage et d'expérimentation**, pas d'un outil professionnel ou commercial.  

⚠️ **Avertissement** : Ce bot ne garantit aucun gain. Les marchés crypto sont volatils et imprévisibles. Vous devez toujours effectuer vos propres analyses avant toute décision d'investissement.

---

## 🎯 Public visé

- **Investisseurs débutants** : Comprendre les signaux du marché  
- **Traders actifs** : Automatiser l'analyse technique  
- **Curieux** : Découvrir l'IA appliquée à la finance  
- **Développeurs** : Étudier une implémentation ML/Crypto  

---

## ⚙️ Prérequis

- **Python 3.11.x** (version recommandée)  
- **RAM** : 4 Go min (8 Go recommandé)  
- **Espace disque** : **50 Go minimum** pour stocker données historiques et modèles  
- **Connexion internet** stable  

Installation des dépendances :  
```bash
pip install -r requirements.txt
```

---

## 🖥️ Configuration personnalisée & Adaptation nécessaire

### ⚡ Bot optimisé pour ma configuration

Ce bot a été **développé et optimisé spécifiquement pour mon environnement de travail** :

- **Configuration matérielle** : Testé uniquement sur ma machine personnelle
- **Ressources système** : Calibré pour mes capacités RAM/CPU spécifiques
- **Chemins et dossiers** : Codés en dur pour mon système Windows
- **Paramètres ML** : Ajustés selon mes performances hardware

### 🔧 Adaptation nécessaire pour d'autres utilisateurs

Si vous souhaitez utiliser ce bot, vous devrez **probablement adapter** :

- **Chemins de fichiers** : Modifier les chemins absolus dans le code
- **Paramètres de mémoire** : Ajuster selon votre RAM disponible
- **Threads/Workers** : Adapter au nombre de cœurs de votre CPU
- **Batch sizes** : Réduire si vous avez moins de mémoire
- **Intervalles de collecte** : Augmenter si connexion plus lente

### 🚀 Version 3 en préparation

Une **V3 est prévue** avec des améliorations majeures :

- ✨ **Auto-configuration** : Détection automatique des ressources système
- 🎯 **Adaptation dynamique** : Ajustement automatique selon votre hardware
- 📦 **Installation simplifiée** : Assistant de configuration intégré
- 🔄 **Profils prédéfinis** : Low/Medium/High selon votre machine
- 🌐 **Multi-plateforme** : Support Windows/Linux/Mac natif

**En attendant** : Cette V2 reste fonctionnelle mais nécessite des ajustements manuels selon votre configuration. N'hésitez pas à ouvrir une issue GitHub si vous rencontrez des difficultés

---

## 🚨 Limitations

- **Pas de trading automatique** – uniquement des prédictions
- **Précision variable** (65–85%) selon la volatilité
- **Max 10 cryptos** suivies
- **Projet d'apprentissage** → Bugs ou imprécisions possibles

---

## 🔍 Fonctionnement

1. **Collecte** des prix, volumes, indicateurs et taux de change
2. **Analyse** via 4 algorithmes IA (Linear Regression, Random Forest, LightGBM, XGBoost)
3. **Prédictions** multi-horizons (10 min à 12 h) avec un score de confiance
4. **Interface CLI** pour lancer, suivre et interroger le bot

---

## 💻 Utilisation rapide

**Lancer le bot** :
```bash
python src/main.py demarrer
```

**Voir l'état du système** :
```bash
python src/main.py etat
```

**Obtenir une prédiction** :
```bash
python src/main.py prevoir
```

---

## 🛡️ Sécurité & Avertissements

- ⛔ **Aucun but commercial**
- ⚠️ **Pas de garantie de résultat**
- 💸 **Ne jamais investir plus que ce que vous pouvez perdre**
- 🔍 **Toujours vérifier les informations avant d'agir**

---

## 📜 Clause légale & Références officielles

Ce projet est distribué sous licence MIT modifiée avec restrictions supplémentaires.

### Clause légale

> Ce code source et ses dérivés sont fournis à titre éducatif et expérimental uniquement.  
> Toute utilisation à des fins commerciales, professionnelles ou par un tiers est strictement interdite sans accord écrit préalable du propriétaire.

### Base légale

Conformément au Code de la propriété intellectuelle français :

**Article L111-1**  
> L'auteur d'une œuvre de l'esprit jouit sur cette œuvre, du seul fait de sa création, d'un droit de propriété incorporelle exclusif et opposable à tous.  
> 🔗 [Lire sur Légifrance](https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006278868/)

**Article L122-4**  
> Toute représentation ou reproduction intégrale ou partielle faite sans le consentement de l'auteur ou de ses ayants droit ou ayants cause est illicite.  
> 🔗 [Lire sur Légifrance](https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006278911/)

### Responsabilité

L'auteur ne pourra être tenu responsable d'aucune perte financière ou dommage résultant de l'utilisation de ce logiciel.

⚠️ **Rappel important** : Les marchés financiers, et en particulier les cryptomonnaies, comportent des risques élevés. Ne jamais investir plus que ce que vous pouvez vous permettre de perdre.

---

## 🐛 Bugs connus & Améliorations prévues

### Problèmes connus
- Décalage horaire possible selon les systèmes
- Première prédiction lente (jusqu'à 1 h)

### Améliorations futures
- Interface web graphique
- Support de plus de cryptos
- Backtesting automatisé
- Notifications push
- Export des prédictions en CSV/Excel
- Création d'un installateur Windows (.exe) pour utilisation comme application PC

---

## 📚 Glossaire rapide

- **Crypto** : Monnaie numérique décentralisée
- **Volatilité** : Amplitude des variations de prix
- **Bull/Bear market** : Marché haussier/baissier
- **HODL** : Conserver à long terme
- **Machine Learning** : IA qui apprend des données
