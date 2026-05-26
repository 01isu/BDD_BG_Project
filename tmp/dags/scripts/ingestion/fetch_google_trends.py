"""
============================================================
fetch_google_trends.py
============================================================
Récupère la popularité Google Trends des sneakers les plus
intéressants depuis le datalake (couche RAW).

Source : pytrends (API non-officielle Google Trends)
Entrée : tmp/data/raw/sneakers/api_sneakers/YYYY-MM-DD.json
Sortie : tmp/data/raw/sneakers/google_trends/YYYY-MM-DD.json
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pytrends.request import TrendReq

# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MAX_KEYWORDS = 25
TIMEFRAME = "today 3-m"
GEO = ""

# Chemins par défaut (à l'intérieur du conteneur Airflow)
INPUT_DIR = Path("/opt/airflow/data/raw/sneakers/api_sneakers")
OUTPUT_DIR = Path("/opt/airflow/data/raw/sneakers/google_trends")


# ============================================================
# Fonctions
# ============================================================

def load_latest_sneakers() -> list[dict]:
    """Charge le dernier fichier de sneakers depuis la couche RAW."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"❌ Le dossier {INPUT_DIR} n'existe pas. "
            "As-tu bien lancé fetch_api_sneakers.py avant ?"
        )

    json_files = sorted(INPUT_DIR.glob("*.json"), reverse=True)
    if not json_files:
        raise FileNotFoundError(
            f"❌ Aucun fichier JSON dans {INPUT_DIR}. "
            "Exécute d'abord fetch_api_sneakers.py"
        )

    latest_file = json_files[0]
    log.info(f"📂 Lecture de : {latest_file.name}")

    with open(latest_file, "r", encoding="utf-8") as f:
        sneakers = json.load(f)

    log.info(f"   ✅ {len(sneakers)} sneakers chargés")
    return sneakers


def build_keywords(sneakers: list[dict]) -> list[dict]:
    """Construit une liste de mots-clés Google Trends à partir des sneakers."""
    keywords = []
    seen = set()

    for sneaker in sneakers:
        sub_series = sneaker.get("sub_series", "").strip()
        nickname = sneaker.get("nickname", "").strip()
        brand = sneaker.get("brand", "").strip()

        if sub_series and nickname:
            keyword = f"{sub_series} {nickname}"
        elif brand and nickname:
            keyword = f"{brand} {nickname}"
        else:
            keyword = sneaker.get("productTitle", "").replace("'", "")

        keyword = keyword.strip()
        if keyword and keyword not in seen and len(keyword) <= 100:
            seen.add(keyword)
            keywords.append({
                "keyword": keyword,
                "sneaker_id": sneaker.get("_id"),
                "product_title": sneaker.get("productTitle"),
                "brand": brand,
                "sub_series": sub_series,
                "nickname": nickname,
            })

    return keywords[:MAX_KEYWORDS]


def fetch_trend_for_keyword(pytrends: TrendReq, kw_info: dict) -> dict:
    """Interroge Google Trends pour 1 mot-clé."""
    keyword = kw_info["keyword"]

    pytrends.build_payload(
        kw_list=[keyword],
        timeframe=TIMEFRAME,
        geo=GEO,
    )

    interest_df = pytrends.interest_over_time()

    if interest_df.empty:
        log.warning(f"   ⚠️ Aucune donnée pour '{keyword}'")
        return None

    points = []
    for date, row in interest_df.iterrows():
        points.append({
            "date": date.isoformat(),
            "value": int(row[keyword]),
        })

    values = [p["value"] for p in points]
    avg_value = sum(values) / len(values) if values else 0
    max_value = max(values) if values else 0

    return {
        **kw_info,
        "trend_points": points,
        "trend_avg": round(avg_value, 2),
        "trend_max": max_value,
        "trend_last": values[-1] if values else 0,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def save_trends(trends_data: list[dict]) -> str:
    """Sauvegarde les tendances Google Trends dans la couche RAW."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = OUTPUT_DIR / f"{today_str}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trends_data, f, ensure_ascii=False, indent=2)

    log.info(f"💾 Sauvegardé : {output_path}")
    return str(output_path)


# ============================================================
# Point d'entrée
# ============================================================

def run():
    """Fonction principale appelée par le DAG Airflow."""
    log.info("=" * 60)
    log.info("🚀 INGESTION Google Trends - DÉMARRAGE")
    log.info("=" * 60)
    log.info(f"📁 INPUT_DIR  = {INPUT_DIR}")
    log.info(f"📁 OUTPUT_DIR = {OUTPUT_DIR}")

    sneakers = load_latest_sneakers()

    keywords = build_keywords(sneakers)
    log.info(f"🎯 {len(keywords)} mots-clés sélectionnés pour Google Trends")

    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=1)

    trends_data = []
    for i, kw_info in enumerate(keywords, 1):
        log.info(f"[{i}/{len(keywords)}] '{kw_info['keyword']}'")
        try:
            result = fetch_trend_for_keyword(pytrends, kw_info)
            if result:
                trends_data.append(result)
            time.sleep(2)
        except Exception as e:
            log.warning(f"   ⚠️ Erreur pour '{kw_info['keyword']}': {e}")
            time.sleep(5)
            continue

    log.info(f"✅ {len(trends_data)} tendances récupérées avec succès")
    output_path = save_trends(trends_data)

    log.info("=" * 60)
    log.info(f"✅ INGESTION Google Trends TERMINÉE")
    log.info("=" * 60)

    return output_path


# ============================================================
# Point d'entrée standalone
# ============================================================
if __name__ == "__main__":
    # Détection automatique : Docker ou local ?
    IN_DOCKER = Path("/opt/airflow").exists()

    if not IN_DOCKER:
        # Test local : adapter les chemins
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
        INPUT_DIR = PROJECT_ROOT / "tmp" / "data" / "raw" / "sneakers" / "api_sneakers"
        OUTPUT_DIR = PROJECT_ROOT / "tmp" / "data" / "raw" / "sneakers" / "google_trends"

    # Sinon (Docker), on garde les chemins /opt/airflow/... définis en haut
    run()