# 👟 Sneakers Resell Tracker — Projet Big Data

Pipeline data end-to-end qui croise les **prix de revente de sneakers** (KicksCrew via RapidAPI) avec leur **popularité** (Google Trends) pour calculer un **Hype Score** et identifier les meilleures opportunités d'achat/revente.

## 🏗️ Architecture

- **Orchestration** : Apache Airflow
- **Datalake** : système de fichiers local (couches `raw / formatted / usage`)
- **Format pivot** : Parquet
- **ML** : Clustering (KMeans) + Régression (RandomForest)
- **Index** : Elasticsearch
- **Dashboard** : Kibana

## 🚀 Lancement

### 1. Prérequis
- Docker Desktop installé et lancé
- Une clé API RapidAPI pour [KicksCrew Sneakers Data](https://rapidapi.com/belchiorarkad-FqvHs2EDOtP/api/kickscrew-sneakers-data)

### 2. Configurer les secrets
Crée un fichier `.env` à la racine en partant de `.env.example` :

```bash
RAPIDAPI_KEY=ta_cle_ici
RAPIDAPI_HOST=kickscrew-sneakers-data.p.rapidapi.com
```

### 3. Démarrer la stack
```bash
docker compose up -d
```

### 4. Accéder aux interfaces
- **Airflow** : http://localhost:8080 (login: `admin` / mot de passe: `admin`)
- **Elasticsearch** : http://localhost:9200
- **Kibana** : http://localhost:5601

### 5. Lancer le pipeline
Dans Airflow, active le DAG `sneakers_pipeline_dag` puis clique sur ▶️ Trigger DAG.

## 📁 Structure du Datalake