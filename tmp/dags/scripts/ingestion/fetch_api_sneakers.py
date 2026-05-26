"""
============================================================
fetch_api_sneakers.py
============================================================
Récupère les données de sneakers depuis l'API KicksCrew
(via RapidAPI) et les sauvegarde dans la couche RAW du datalake.

Source : https://rapidapi.com/belchiorarkad-FqvHs2EDOtP/api/kickscrew-sneakers-data
Endpoint utilisé : GET /search?query=<brand>
Sortie : tmp/data/raw/sneakers/api_sneakers/YYYY-MM-DD.json
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BRANDS_TO_FETCH = [
    "yeezy",
    "nike dunk",
    "jordan",
    "new balance",
    "adidas samba",
    "asics",
]

RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "kickscrew-sneakers-data.p.rapidapi.com")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
API_URL = f"https://{RAPIDAPI_HOST}/search"

ALLOWED_PRODUCT_TYPES = {"Sneakers", "Slippers", "Sandals"}

# Chemin de sortie (à l'intérieur du conteneur Airflow)
OUTPUT_DIR = Path("/opt/airflow/data/raw/sneakers/api_sneakers")


# ============================================================
# Fonctions
# ============================================================

def fetch_brand(brand: str) -> list[dict]:
    """Interroge l'API KicksCrew pour une marque donnée."""
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "Content-Type": "application/json",
    }
    params = {"query": brand}

    log.info(f"📡 Requête API pour '{brand}'...")
    response = requests.get(API_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    products = data.get("products", [])
    log.info(f"   ✅ {len(products)} produits reçus pour '{brand}'")
    return products


def filter_products(products: list[dict]) -> list[dict]:
    """Garde uniquement les chaussures (filtre hoodies/t-shirts/etc.)."""
    filtered = []
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    for product in products:
        product_type = product.get("product_type", "")
        if product_type in ALLOWED_PRODUCT_TYPES:
            product["fetched_at"] = timestamp_utc
            filtered.append(product)

    return filtered


def deduplicate(products: list[dict]) -> list[dict]:
    """Supprime les doublons basés sur l'_id du produit."""
    seen_ids = set()
    unique_products = []

    for product in products:
        pid = product.get("_id")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_products.append(product)

    return unique_products


def save_to_datalake(products: list[dict]) -> str:
    """Sauvegarde la liste des produits en JSON dans la couche RAW."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = OUTPUT_DIR / f"{today_str}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    log.info(f"💾 Sauvegardé : {output_path}")
    log.info(f"   📊 {len(products)} produits dans le fichier")
    return str(output_path)


# ============================================================
# Point d'entrée
# ============================================================

def run():
    """Fonction principale appelée par le DAG Airflow."""
    log.info("=" * 60)
    log.info("🚀 INGESTION KicksCrew - DÉMARRAGE")
    log.info("=" * 60)
    log.info(f"📁 OUTPUT_DIR = {OUTPUT_DIR}")

    if not RAPIDAPI_KEY:
        raise ValueError(
            "❌ La variable d'environnement RAPIDAPI_KEY n'est pas définie. "
            "Vérifie ton fichier .env"
        )

    all_products = []

    for brand in BRANDS_TO_FETCH:
        try:
            products = fetch_brand(brand)
            all_products.extend(products)
            time.sleep(1)
        except requests.HTTPError as e:
            log.error(f"⚠️ Erreur API pour '{brand}': {e}")
            continue
        except Exception as e:
            log.error(f"⚠️ Erreur inattendue pour '{brand}': {e}")
            continue

    log.info(f"📦 Total brut récupéré : {len(all_products)} produits")

    filtered = filter_products(all_products)
    log.info(f"👟 Après filtre chaussures : {len(filtered)} produits")

    unique = deduplicate(filtered)
    log.info(f"🎯 Après déduplication : {len(unique)} produits uniques")

    output_path = save_to_datalake(unique)

    log.info("=" * 60)
    log.info(f"✅ INGESTION TERMINÉE — {len(unique)} sneakers")
    log.info("=" * 60)

    return output_path


# ============================================================
# Point d'entrée standalone
# ============================================================
if __name__ == "__main__":
    # Détection automatique : Docker ou local ?
    IN_DOCKER = Path("/opt/airflow").exists()

    if not IN_DOCKER:
        # Test local : adapter les chemins et charger .env
        try:
            from dotenv import load_dotenv
            project_root = Path(__file__).parent.parent.parent.parent.parent
            load_dotenv(project_root / ".env")
        except ImportError:
            pass

        OUTPUT_DIR = (
            Path(__file__).parent.parent.parent.parent
            / "data" / "raw" / "sneakers" / "api_sneakers"
        )

    # Sinon (Docker), on garde le chemin /opt/airflow/... défini en haut
    run()