"""
============================================================
index_to_elastic.py
============================================================
Indexe le Parquet final de la couche USAGE dans Elasticsearch
pour visualisation Kibana.

Entrée :
  - tmp/data/usage/sneakers_hype/YYYY-MM-DD.parquet

Sortie :
  - Index Elasticsearch 'sneakers_hype' (réécrit à chaque run)
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

USAGE_DIR = Path("/opt/airflow/data/usage/sneakers_hype")

# URL Elasticsearch (depuis le conteneur Airflow, on accède via le nom du service)
ELASTIC_HOST = os.getenv("ELASTIC_HOST", "http://elasticsearch:9200")

# Nom de l'index dans Elasticsearch
INDEX_NAME = "sneakers_hype"

# Mapping = définition du schéma de l'index dans Elasticsearch
# C'est l'équivalent du "CREATE TABLE" en SQL
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            # Identifiants
            "sneaker_id": {"type": "keyword"},
            "parent_product_id": {"type": "keyword"},
            "model_no": {"type": "keyword"},

            # Identité (texte cherchable + keyword pour aggregations)
            "product_title": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "brand": {"type": "keyword"},
            "series": {"type": "keyword"},
            "sub_series": {"type": "keyword"},
            "nickname": {"type": "keyword"},
            "colorway": {"type": "text"},
            "color": {"type": "keyword"},
            "product_type": {"type": "keyword"},
            "gender": {"type": "keyword"},
            "season": {"type": "keyword"},

            # Prix
            "price": {"type": "float"},
            "lowest_price": {"type": "float"},
            "lowest_price_original": {"type": "float"},
            "lowest_instantship_price": {"type": "float"},
            "highest_price": {"type": "float"},

            # Tailles
            "lowest_price_size": {"type": "keyword"},
            "highest_price_size": {"type": "keyword"},
            "inventory_sizes_count": {"type": "integer"},

            # Dates
            "release_date": {"type": "date"},
            "release_year": {"type": "integer"},
            "updated_at": {"type": "date"},
            "fetched_at": {"type": "date"},

            # Booléens
            "has_instantship": {"type": "boolean"},
            "has_image": {"type": "boolean"},
            "is_new_release": {"type": "boolean"},
            "on_sale": {"type": "boolean"},
            "any_variant_available": {"type": "boolean"},
            "has_google_trend": {"type": "boolean"},
            "is_grail": {"type": "boolean"},

            # KPIs
            "discount_pct": {"type": "float"},
            "markup_pct": {"type": "float"},
            "age_days": {"type": "integer"},

            # Google Trends
            "trend_avg": {"type": "float"},
            "trend_max": {"type": "integer"},
            "trend_last": {"type": "integer"},
            "keyword": {"type": "keyword"},

            # Hype Score
            "score_popularity": {"type": "float"},
            "score_valuation": {"type": "float"},
            "score_opportunity": {"type": "float"},
            "hype_score": {"type": "float"},

            # ML
            "cluster_id": {"type": "integer"},
            "cluster_label": {"type": "keyword"},
            "predicted_price": {"type": "float"},
            "price_gap_pct": {"type": "float"},

            # Image
            "image_url": {"type": "keyword", "index": False},

            # Description
            "description": {"type": "text"},
        }
    }
}


# ============================================================
# Helpers
# ============================================================

def get_latest_file(directory: Path, extension: str = "*.parquet") -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"❌ Dossier introuvable : {directory}")
    files = sorted(directory.glob(extension), reverse=True)
    if not files:
        raise FileNotFoundError(f"❌ Aucun fichier {extension} dans {directory}")
    return files[0]


def clean_value(v):
    """
    Nettoie une valeur pour Elasticsearch :
    - NaN / NaT / None → null
    - Timestamps pandas → ISO string
    - numpy types → types Python natifs
    """
    if v is None:
        return None
    # pandas NaT (Not a Time)
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return None
        return v.isoformat()
    # Floats / numbers
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    # pandas NA
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


# Liste des champs qui doivent être de vrais booléens dans Elasticsearch
BOOLEAN_FIELDS = {
    "has_instantship",
    "has_image",
    "is_new_release",
    "on_sale",
    "any_variant_available",
    "has_google_trend",
    "is_grail",
}


def row_to_doc(row: pd.Series) -> dict:
    """Convertit une ligne du DataFrame en document Elasticsearch propre."""
    doc = {}
    for key, val in row.items():
        cleaned = clean_value(val)
        # Conversion explicite des champs booléens (0/1 → False/True)
        if key in BOOLEAN_FIELDS and cleaned is not None:
            cleaned = bool(cleaned)
        doc[key] = cleaned
    return doc


# ============================================================
# Indexing
# ============================================================

def connect_elastic() -> Elasticsearch:
    """Se connecte à Elasticsearch et vérifie la connexion."""
    log.info(f"🔌 Connexion à Elasticsearch : {ELASTIC_HOST}")
    es = Elasticsearch(ELASTIC_HOST, request_timeout=30)

    if not es.ping():
        raise ConnectionError(f"❌ Elasticsearch injoignable à {ELASTIC_HOST}")

    info = es.info()
    log.info(f"   ✅ Connecté — Elasticsearch v{info['version']['number']}")
    return es


def recreate_index(es: Elasticsearch):
    """Supprime l'index s'il existe et le recrée avec le bon mapping."""
    if es.indices.exists(index=INDEX_NAME):
        log.info(f"🗑️  Suppression de l'ancien index '{INDEX_NAME}'")
        es.indices.delete(index=INDEX_NAME)

    log.info(f"📝 Création de l'index '{INDEX_NAME}' avec mapping...")
    es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    log.info(f"   ✅ Index créé")


def index_dataframe(es: Elasticsearch, df: pd.DataFrame):
    """Indexe tout le DataFrame en bulk dans Elasticsearch."""
    log.info(f"📤 Indexation de {len(df)} documents...")

    actions = []
    for _, row in df.iterrows():
        doc = row_to_doc(row)
        actions.append({
            "_index": INDEX_NAME,
            "_id": doc.get("sneaker_id"),  # utilise sneaker_id comme clé primaire
            "_source": doc,
        })

    success, errors = bulk(es, actions, raise_on_error=False)
    log.info(f"   ✅ {success} documents indexés avec succès")

    if errors:
        log.warning(f"   ⚠️ {len(errors)} erreurs d'indexation")
        for err in errors[:3]:
            log.warning(f"      - {err}")

    # Refresh pour rendre les documents tout de suite cherchables
    es.indices.refresh(index=INDEX_NAME)

    count = es.count(index=INDEX_NAME)["count"]
    log.info(f"📊 Total dans l'index : {count} documents")


# ============================================================
# Point d'entrée principal
# ============================================================

def run():
    """Fonction principale appelée par le DAG Airflow."""
    log.info("=" * 60)
    log.info("🚀 INDEXING — Usage → Elasticsearch")
    log.info("=" * 60)

    # 1. Lecture du Parquet final
    usage_path = get_latest_file(USAGE_DIR)
    log.info(f"📂 Lecture : {usage_path.name}")
    df = pd.read_parquet(usage_path)
    log.info(f"   ✅ {len(df)} sneakers chargés depuis le datalake")

    # 2. Connexion Elasticsearch
    es = connect_elastic()

    # 3. Recréation de l'index (mapping propre)
    recreate_index(es)

    # 4. Indexation
    index_dataframe(es, df)

    log.info("=" * 60)
    log.info(f"✅ INDEXING TERMINÉ — Données prêtes pour Kibana")
    log.info(f"🌐 Accède à http://localhost:5601")
    log.info("=" * 60)


# ============================================================
# Standalone (test local hors Docker)
# ============================================================
if __name__ == "__main__":
    IN_DOCKER = Path("/opt/airflow").exists()

    if not IN_DOCKER:
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
        USAGE_DIR = PROJECT_ROOT / "tmp" / "data" / "usage" / "sneakers_hype"
        ELASTIC_HOST = "http://localhost:9200"

    run()