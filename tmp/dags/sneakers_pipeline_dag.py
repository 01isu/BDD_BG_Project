"""
============================================================
sneakers_pipeline_dag.py
============================================================
DAG Airflow qui orchestre le pipeline complet du projet
"Sneakers Resell Tracker".

Pipeline :
  1. fetch_api_sneakers   : récupère KicksCrew via RapidAPI
  2. fetch_google_trends  : récupère Google Trends
  3. format_to_parquet    : JSON brut → Parquet typé UTC
  4. combine_sources      : jointure + Hype Score + ML
  5. index_to_elastic     : push final vers Elasticsearch

Schedule : quotidien à 6h UTC (la source la plus fraîche)
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ============================================================
# Imports des fonctions run() de chaque script
# ============================================================
# On ajoute le dossier scripts/ au sys.path pour les imports
import sys
SCRIPTS_DIR = Path("/opt/airflow/dags/scripts")
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ingestion.fetch_api_sneakers import run as run_fetch_sneakers
from ingestion.fetch_google_trends import run as run_fetch_trends
from formatting.format_to_parquet import run as run_format
from combination.combine_sources import run as run_combine
from indexing.index_to_elastic import run as run_index


# ============================================================
# Configuration du DAG
# ============================================================

default_args = {
    "owner": "louis",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="sneakers_pipeline",
    description="Pipeline Big Data : sneakers resell tracker avec Hype Score + ML",
    default_args=default_args,
    schedule="0 6 * * *",  # tous les jours à 6h UTC (cron format)
    start_date=datetime(2026, 5, 1),
    catchup=False,  # pas de rattrapage des jours passés
    max_active_runs=1,  # une seule exécution à la fois
    tags=["sneakers", "bigdata", "ml", "isep"],
) as dag:

    # =====================================================
    # Tâche 1 : Ingestion KicksCrew
    # =====================================================
    task_fetch_sneakers = PythonOperator(
        task_id="fetch_api_sneakers",
        python_callable=run_fetch_sneakers,
        doc_md="""
        ### Ingestion KicksCrew
        Récupère ~80-100 sneakers depuis l'API KicksCrew via RapidAPI.
        Sortie : `tmp/data/raw/sneakers/api_sneakers/YYYY-MM-DD.json`
        """,
    )

    # =====================================================
    # Tâche 2 : Ingestion Google Trends
    # =====================================================
    task_fetch_trends = PythonOperator(
        task_id="fetch_google_trends",
        python_callable=run_fetch_trends,
        doc_md="""
        ### Ingestion Google Trends
        Récupère la popularité Google Trends des 25 sneakers les plus
        prometteuses identifiées à l'étape précédente.
        Sortie : `tmp/data/raw/sneakers/google_trends/YYYY-MM-DD.json`
        """,
    )

    # =====================================================
    # Tâche 3 : Formatting → Parquet
    # =====================================================
    task_format = PythonOperator(
        task_id="format_to_parquet",
        python_callable=run_format,
        doc_md="""
        ### Formatting (Raw → Formatted)
        Convertit les JSON bruts en Parquet typé avec :
        - Dates normalisées en UTC
        - Prix en float
        - KPIs dérivés (discount %, markup %, age days)
        Sortie : `tmp/data/formatted/sneakers/<source>/YYYY-MM-DD.parquet`
        """,
    )

    # =====================================================
    # Tâche 4 : Combination + ML
    # =====================================================
    task_combine = PythonOperator(
        task_id="combine_sources_and_ml",
        python_callable=run_combine,
        doc_md="""
        ### Combination + Machine Learning
        - Jointure KicksCrew × Google Trends
        - Calcul du Sneaker Hype Score (0-100)
        - KMeans clustering (4 personas + catégorie Grail)
        - RandomForest pour prédiction de prix
        Sortie : `tmp/data/usage/sneakers_hype/YYYY-MM-DD.parquet`
        """,
    )

    # =====================================================
    # Tâche 5 : Indexing Elasticsearch
    # =====================================================
    task_index = PythonOperator(
        task_id="index_to_elasticsearch",
        python_callable=run_index,
        doc_md="""
        ### Indexing dans Elasticsearch
        Recrée l'index `sneakers_hype` avec un mapping strict et
        indexe tous les documents pour visualisation dans Kibana.
        Sortie : index Elasticsearch `sneakers_hype` (consultable sur http://localhost:5601)
        """,
    )

    # =====================================================
    # Définition de l'ordre d'exécution
    # =====================================================
    # Les 2 ingestions peuvent tourner en séquence
    # (fetch_trends dépend du fichier produit par fetch_sneakers)
    (
        task_fetch_sneakers
        >> task_fetch_trends
        >> task_format
        >> task_combine
        >> task_index
    )