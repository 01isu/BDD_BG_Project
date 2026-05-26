"""
============================================================
format_to_parquet.py
============================================================
Convertit les JSON bruts (couche RAW) en fichiers Parquet
propres et normalisés (couche FORMATTED).

Transformations appliquées :
  - Sélection des colonnes utiles
  - Typage strict (int, float, string, datetime)
  - Normalisation des dates en UTC
  - Gestion des valeurs nulles
  - Aplatissement des listes (image_urls, colorway...)

Entrée :
  - tmp/data/raw/sneakers/api_sneakers/YYYY-MM-DD.json
  - tmp/data/raw/sneakers/google_trends/YYYY-MM-DD.json

Sortie :
  - tmp/data/formatted/sneakers/api_sneakers/YYYY-MM-DD.parquet
  - tmp/data/formatted/sneakers/google_trends/YYYY-MM-DD.parquet
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Chemins (à l'intérieur du conteneur Airflow)
RAW_DIR = Path("/opt/airflow/data/raw/sneakers")
FORMATTED_DIR = Path("/opt/airflow/data/formatted/sneakers")


# ============================================================
# Helpers
# ============================================================

def get_latest_file(directory: Path, extension: str = "*.json") -> Path:
    """Retourne le fichier le plus récent dans un dossier."""
    if not directory.exists():
        raise FileNotFoundError(f"❌ Dossier introuvable : {directory}")

    files = sorted(directory.glob(extension), reverse=True)
    if not files:
        raise FileNotFoundError(f"❌ Aucun fichier {extension} dans {directory}")

    return files[0]


def unix_ms_to_utc(unix_ms) -> pd.Timestamp | None:
    """Convertit un timestamp Unix en millisecondes vers datetime UTC."""
    if pd.isna(unix_ms) or unix_ms == 0:
        return None
    try:
        # Si déjà en secondes (10 chiffres), on multiplie par 1000
        if unix_ms < 1e11:
            unix_ms = unix_ms * 1000
        return pd.to_datetime(unix_ms, unit="ms", utc=True)
    except (ValueError, TypeError):
        return None


def yyyymmdd_to_utc(yyyymmdd) -> pd.Timestamp | None:
    """Convertit un entier YYYYMMDD (ex: 20220306) en datetime UTC."""
    if pd.isna(yyyymmdd) or yyyymmdd == 0:
        return None
    try:
        return pd.to_datetime(str(int(yyyymmdd)), format="%Y%m%d", utc=True)
    except (ValueError, TypeError):
        return None


def safe_first(lst) -> str | None:
    """Retourne le 1er élément d'une liste, ou None si vide."""
    if isinstance(lst, list) and lst:
        return lst[0]
    return None


def safe_join(lst, sep: str = ", ") -> str | None:
    """Joint une liste de chaînes (avec déduplication)."""
    if isinstance(lst, list) and lst:
        # Set pour dédupliquer, puis trier pour reproductibilité
        return sep.join(sorted(set(str(x) for x in lst if x)))
    return None


# ============================================================
# Formatting des sneakers (KicksCrew)
# ============================================================

def format_sneakers(input_path: Path) -> pd.DataFrame:
    """Charge le JSON KicksCrew et le transforme en DataFrame propre."""
    log.info(f"📂 Lecture : {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    log.info(f"   ✅ {len(raw_data)} sneakers bruts chargés")

    # On construit un DataFrame en sélectionnant les colonnes intéressantes
    rows = []
    for item in raw_data:
        rows.append({
            # Identifiants
            "sneaker_id": str(item.get("_id", "")),
            "parent_product_id": str(item.get("parentProductId", "")),
            "model_no": item.get("model_no"),

            # Identité
            "product_title": item.get("productTitle"),
            "brand": item.get("brand"),
            "series": item.get("series"),
            "sub_series": item.get("sub_series"),
            "nickname": item.get("nickname"),
            "colorway": safe_join(item.get("colorway")),
            "color": item.get("color"),
            "product_type": item.get("product_type"),
            "gender": item.get("gender"),
            "season": item.get("season"),

            # Prix (les vrais nombres en float)
            "price": item.get("price"),
            "lowest_price": item.get("lowest_price"),
            "lowest_price_original": item.get("lowest_price_original"),
            "lowest_instantship_price": item.get("lowest_instantship_price"),
            "highest_price": item.get("highest_price"),

            # Tailles
            "lowest_price_size": item.get("lowest_price_size"),
            "highest_price_size": item.get("highest_price_size"),
            "inventory_sizes_count": len(item.get("allInventory", [])),

            # Dates (à parser en UTC)
            "release_date_raw": item.get("release_date"),
            "release_date_unix_raw": item.get("release_date_unix"),
            "release_year": item.get("release_year"),
            "updated_at_raw": item.get("updated_at"),
            "fetched_at_raw": item.get("fetched_at"),

            # Indicateurs business
            "has_instantship": bool(item.get("has_instantship", False)),
            "has_image": bool(item.get("has_image", False)),
            "is_new_release": bool(item.get("is_new_release", False)),
            "on_sale": bool(item.get("on_sale", False)),
            "any_variant_available": bool(item.get("anyVariantInventoryAvailable", False)),

            # Image principale (juste la 1ère, c'est suffisant pour Kibana)
            "image_url": safe_first(item.get("image_urls")),

            # Description (utile pour explorer / debug)
            "description": item.get("description"),
        })

    df = pd.DataFrame(rows)
    log.info(f"   📊 DataFrame initial : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # ========================================
    # Typage et normalisation
    # ========================================

    # Prix : tout en float (Pandas convertit None → NaN)
    price_cols = [
        "price", "lowest_price", "lowest_price_original",
        "lowest_instantship_price", "highest_price",
    ]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # release_year : int (gère NaN avec Int64)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

    # Dates UTC
    df["release_date"] = df["release_date_raw"].apply(yyyymmdd_to_utc)
    df["release_date_from_unix"] = df["release_date_unix_raw"].apply(unix_ms_to_utc)
    df["updated_at"] = df["updated_at_raw"].apply(unix_ms_to_utc)
    df["fetched_at"] = pd.to_datetime(df["fetched_at_raw"], utc=True, errors="coerce")

    # On supprime les colonnes "raw" devenues inutiles
    df = df.drop(columns=[
        "release_date_raw", "release_date_unix_raw",
        "updated_at_raw", "fetched_at_raw",
    ])

    # KPIs dérivés (super utiles pour Kibana et le ML)
    # Discount = entre lowest_price et highest_price
    df["discount_pct"] = (
        (df["highest_price"] - df["lowest_price"]) / df["highest_price"] * 100
    ).round(2)

    # Markup = entre prix d'origine et prix actuel (positif = sneaker valorisée)
    df["markup_pct"] = (
        (df["lowest_price"] - df["lowest_price_original"]) / df["lowest_price_original"] * 100
    ).round(2)

    # Âge en jours depuis la sortie
    now_utc = pd.Timestamp.now(tz="UTC")
    df["age_days"] = (now_utc - df["release_date"]).dt.days.astype("Int64")

    log.info(f"   ✅ Normalisation OK : {df.shape[1]} colonnes finales")
    return df


# ============================================================
# Formatting des Google Trends
# ============================================================

def format_trends(input_path: Path) -> pd.DataFrame:
    """Charge le JSON Google Trends et le transforme en DataFrame propre."""
    log.info(f"📂 Lecture : {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    log.info(f"   ✅ {len(raw_data)} tendances brutes chargées")

    rows = []
    for item in raw_data:
        rows.append({
            "sneaker_id": str(item.get("sneaker_id", "")),
            "keyword": item.get("keyword"),
            "product_title": item.get("product_title"),
            "brand": item.get("brand"),
            "sub_series": item.get("sub_series"),
            "nickname": item.get("nickname"),
            "trend_avg": item.get("trend_avg"),
            "trend_max": item.get("trend_max"),
            "trend_last": item.get("trend_last"),
            "trend_points_count": len(item.get("trend_points", [])),
            "fetched_at_raw": item.get("fetched_at"),
        })

    df = pd.DataFrame(rows)

    # Typage
    df["trend_avg"] = pd.to_numeric(df["trend_avg"], errors="coerce")
    df["trend_max"] = pd.to_numeric(df["trend_max"], errors="coerce").astype("Int64")
    df["trend_last"] = pd.to_numeric(df["trend_last"], errors="coerce").astype("Int64")
    df["fetched_at"] = pd.to_datetime(df["fetched_at_raw"], utc=True, errors="coerce")
    df = df.drop(columns=["fetched_at_raw"])

    log.info(f"   ✅ Normalisation OK : {df.shape[1]} colonnes finales")
    return df


# ============================================================
# Sauvegarde Parquet
# ============================================================

def save_parquet(df: pd.DataFrame, output_dir: Path, source_name: str) -> Path:
    """Sauvegarde un DataFrame en Parquet avec date d'aujourd'hui."""
    output_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = output_dir / f"{today_str}.parquet"

    df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

    # Stats pour le log
    size_kb = output_path.stat().st_size / 1024
    log.info(f"💾 [{source_name}] Sauvegardé : {output_path}")
    log.info(f"   📊 {df.shape[0]} lignes × {df.shape[1]} colonnes — {size_kb:.1f} Ko")
    return output_path


# ============================================================
# Point d'entrée principal
# ============================================================

def run():
    """Fonction principale appelée par le DAG Airflow."""
    log.info("=" * 60)
    log.info("🚀 FORMATTING — Raw → Formatted (Parquet)")
    log.info("=" * 60)

    # ---------- KicksCrew ----------
    log.info("--- KicksCrew Sneakers ---")
    sneakers_raw = get_latest_file(RAW_DIR / "api_sneakers")
    df_sneakers = format_sneakers(sneakers_raw)
    save_parquet(df_sneakers, FORMATTED_DIR / "api_sneakers", "Sneakers")

    # ---------- Google Trends ----------
    log.info("--- Google Trends ---")
    trends_raw = get_latest_file(RAW_DIR / "google_trends")
    df_trends = format_trends(trends_raw)
    save_parquet(df_trends, FORMATTED_DIR / "google_trends", "Trends")

    log.info("=" * 60)
    log.info("✅ FORMATTING TERMINÉ — Couche FORMATTED prête")
    log.info("=" * 60)


# ============================================================
# Standalone (test local hors Docker)
# ============================================================
if __name__ == "__main__":
    IN_DOCKER = Path("/opt/airflow").exists()

    if not IN_DOCKER:
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
        RAW_DIR = PROJECT_ROOT / "tmp" / "data" / "raw" / "sneakers"
        FORMATTED_DIR = PROJECT_ROOT / "tmp" / "data" / "formatted" / "sneakers"

    run()