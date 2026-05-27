"""
============================================================
combine_sources.py
============================================================
Combine les sneakers KicksCrew avec leurs tendances Google,
calcule le "Sneaker Hype Score" et applique 2 modèles ML :
  - KMeans : segmentation business (4 clusters + catégorie Grail)
  - RandomForest : prédiction de prix sans data leakage

Entrée :
  - tmp/data/formatted/sneakers/api_sneakers/YYYY-MM-DD.parquet
  - tmp/data/formatted/sneakers/google_trends/YYYY-MM-DD.parquet

Sortie :
  - tmp/data/usage/sneakers_hype/YYYY-MM-DD.parquet
"""

import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

FORMATTED_DIR = Path("/opt/airflow/data/formatted/sneakers")
USAGE_DIR = Path("/opt/airflow/data/usage/sneakers_hype")

N_CLUSTERS = 4


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


# ============================================================
# 1. Chargement et jointure
# ============================================================

def load_and_join() -> pd.DataFrame:
    """Charge les 2 sources Parquet et les joint sur sneaker_id."""
    sneakers_path = get_latest_file(FORMATTED_DIR / "api_sneakers")
    trends_path = get_latest_file(FORMATTED_DIR / "google_trends")

    log.info(f"📂 Lecture sneakers : {sneakers_path.name}")
    df_sneakers = pd.read_parquet(sneakers_path)
    log.info(f"   ✅ {len(df_sneakers)} sneakers chargés")

    log.info(f"📂 Lecture trends : {trends_path.name}")
    df_trends = pd.read_parquet(trends_path)
    log.info(f"   ✅ {len(df_trends)} tendances chargées")

    df_trends_subset = df_trends[
        ["sneaker_id", "trend_avg", "trend_max", "trend_last", "keyword"]
    ].copy()

    df = df_sneakers.merge(df_trends_subset, on="sneaker_id", how="left")

    df["trend_avg"] = df["trend_avg"].fillna(0)
    df["trend_max"] = df["trend_max"].fillna(0).astype("Int64")
    df["trend_last"] = df["trend_last"].fillna(0).astype("Int64")
    df["has_google_trend"] = df["keyword"].notna()

    with_trend = df["has_google_trend"].sum()
    log.info(f"🔗 Jointure : {len(df)} sneakers — {with_trend} avec tendance Google")

    return df


# ============================================================
# 2. Marquage des sneakers Grails (ultra-rares)
# ============================================================

def mark_grails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marque les sneakers ultra-rares (top 5% en prix) comme 'Grails'.
    Ces sneakers sont gardées dans le dataset mais isolées du clustering.
    """
    grail_threshold = df["lowest_price"].quantile(0.95)

    df["is_grail"] = (df["lowest_price"] > grail_threshold).astype(int)
    n_grails = df["is_grail"].sum()

    log.info(
        f"💎 Marquage Grails : {n_grails} sneakers ultra-rares "
        f"(prix > ${grail_threshold:.0f})"
    )

    if n_grails > 0:
        log.info("   Sneakers Grails identifiées :")
        grails = df[df["is_grail"] == 1].nlargest(5, "lowest_price")
        for _, row in grails.iterrows():
            log.info(f"      - {row['product_title']} : ${row['lowest_price']:.0f}")

    return df


# ============================================================
# 3. Calcul du Sneaker Hype Score
# ============================================================

def compute_hype_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le Sneaker Hype Score (0-100).
    Formule : popularité (50%) + valorisation (30%) + opportunité (20%)
    """
    log.info("🧮 Calcul du Sneaker Hype Score...")

    df["score_popularity"] = df["trend_avg"].clip(0, 100)

    markup = df["markup_pct"].fillna(0)
    df["score_valuation"] = markup.clip(lower=0, upper=200) / 2

    discount = df["discount_pct"].fillna(0)
    df["score_opportunity"] = discount.clip(0, 100)

    df["hype_score"] = (
        0.50 * df["score_popularity"]
        + 0.30 * df["score_valuation"]
        + 0.20 * df["score_opportunity"]
    ).round(2)

    log.info(f"   ✅ Hype Score calculé — moyenne : {df['hype_score'].mean():.1f}/100")
    log.info("   🥇 Top 3 :")
    top3 = df.nlargest(3, "hype_score")[["product_title", "hype_score"]]
    for _, row in top3.iterrows():
        log.info(f"      - {row['product_title']} → {row['hype_score']}/100")

    return df


# ============================================================
# 4. KMeans clustering (Grails isolés)
# ============================================================

def apply_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmente les sneakers en clusters :
    - Les Grails (cluster -1) sont assignés à part
    - KMeans s'occupe du reste sans être perturbé par les outliers
    """
    log.info(f"🤖 KMeans clustering ({N_CLUSTERS} clusters + Grails)...")

    df["cluster_id"] = -1
    df["cluster_label"] = None

    grails_mask = df["is_grail"] == 1
    df.loc[grails_mask, "cluster_id"] = -1
    df.loc[grails_mask, "cluster_label"] = "👑 Grail (Ultra-Rare)"

    df_normal = df[~grails_mask].copy()
    log.info(f"   📊 KMeans sur {len(df_normal)} sneakers (hors Grails)")

    features = pd.DataFrame(index=df_normal.index)
    features["log_price"] = np.log1p(
        df_normal["lowest_price"].fillna(df_normal["lowest_price"].median())
    )
    features["trend_avg"] = df_normal["trend_avg"].fillna(0)
    features["age_years"] = (
        df_normal["age_days"].fillna(df_normal["age_days"].median()) / 365
    )
    features["discount_pct"] = df_normal["discount_pct"].fillna(0)

    log.info(f"   📊 Features KMeans : {list(features.columns)}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    cluster_ids = kmeans.fit_predict(features_scaled)

    df.loc[df_normal.index, "cluster_id"] = cluster_ids

    centroids_real = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_real, columns=features.columns)
    centroids_df["price_real"] = np.expm1(centroids_df["log_price"])

    centroids_df["business_score"] = (
        centroids_df["trend_avg"] * 3 + centroids_df["discount_pct"] * 0.5
    )

    centroids_sorted = centroids_df.sort_values("business_score", ascending=False)
    sorted_cluster_ids = centroids_sorted.index.tolist()

    rank_labels = [
        "🔥 Holy Grail",
        "🚀 Hype Bargain",
        "💎 Sleeper Pick",
        "📦 Mass Market",
    ]

    cluster_labels = {}
    for rank, cluster_id in enumerate(sorted_cluster_ids):
        cluster_labels[cluster_id] = (
            rank_labels[rank] if rank < len(rank_labels) else f"Cluster {cluster_id}"
        )

    df.loc[~grails_mask, "cluster_label"] = (
        df.loc[~grails_mask, "cluster_id"].map(cluster_labels)
    )

    log.info("   ✅ Clusters identifiés :")
    n_grails = int(grails_mask.sum())
    log.info(f"      Cluster -1 (👑 Grail Ultra-Rare) : {n_grails} sneakers")
    for cluster_id in range(N_CLUSTERS):
        label = cluster_labels[cluster_id]
        count = int(((df["cluster_id"] == cluster_id) & (~grails_mask)).sum())
        c = centroids_df.iloc[cluster_id]
        log.info(
            f"      Cluster {cluster_id} ({label}) : {count} sneakers — "
            f"prix médian ~${c['price_real']:.0f}, "
            f"trend ~{c['trend_avg']:.1f}, "
            f"discount ~{c['discount_pct']:.1f}%, "
            f"âge ~{c['age_years']:.1f}ans"
        )

    return df


# ============================================================
# 5. RandomForest (sans data leakage, sans Grails)
# ============================================================

def apply_random_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entraîne un RandomForest pour prédire lowest_price avec des
    features indépendantes du prix. Exclut les Grails (outliers).
    """
    log.info("🌲 RandomForest pour prédiction de prix...")

    # On entraîne SANS les Grails (ils faussent l'apprentissage)
    df_train = df[df["is_grail"] == 0].copy()
    log.info(f"   📊 Dataset d'entraînement : {len(df_train)} sneakers (Grails exclus)")

    feature_cols = [
        "release_year",
        "age_days",
        "trend_avg",
        "trend_max",
        "trend_last",
        "inventory_sizes_count",
        "has_instantship",
        "on_sale",
        "is_new_release",
    ]

    target_col = "lowest_price"

    df_ml_full = df_train[feature_cols + [target_col]].copy()
    for bool_col in ["has_instantship", "on_sale", "is_new_release"]:
        df_ml_full[bool_col] = df_ml_full[bool_col].astype(int)

    df_ml = df_ml_full.dropna(subset=[target_col])

    if len(df_ml) < 20:
        log.warning(f"   ⚠️ Pas assez de données ({len(df_ml)} lignes), ML sauté.")
        df["predicted_price"] = np.nan
        df["price_gap_pct"] = np.nan
        return df

    medians = {col: df_ml[col].median() for col in feature_cols}
    for col in feature_cols:
        df_ml[col] = df_ml[col].fillna(medians[col])

    log.info(f"   📊 Dataset ML final : {len(df_ml)} lignes × {len(feature_cols)} features")

    X = df_ml[feature_cols]
    y = df_ml[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    log.info(f"   ✅ Performance — MAE: ${mae:.2f} | R²: {r2:.3f}")

    # Prédiction sur TOUT le dataset (Grails inclus)
    df_predict = df[feature_cols].copy()
    for bool_col in ["has_instantship", "on_sale", "is_new_release"]:
        df_predict[bool_col] = df_predict[bool_col].astype(int)
    for col in feature_cols:
        df_predict[col] = df_predict[col].fillna(medians[col])

    df["predicted_price"] = rf.predict(df_predict).round(2)

    df["price_gap_pct"] = (
        (df["lowest_price"] - df["predicted_price"]) / df["predicted_price"] * 100
    ).round(2)

    # On affiche les opportunités UNIQUEMENT parmi les sneakers normales
    undervalued_mask = (df["price_gap_pct"] < -20) & (df["is_grail"] == 0)
    undervalued = df[undervalued_mask].nsmallest(3, "price_gap_pct")
    if len(undervalued) > 0:
        log.info("   💡 Top 3 sneakers sous-évaluées :")
        for _, row in undervalued.iterrows():
            log.info(
                f"      - {row['product_title']} : "
                f"${row['lowest_price']:.0f} (prédit ${row['predicted_price']:.0f}) "
                f"→ {row['price_gap_pct']}%"
            )

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    log.info("   📈 Top 3 features les plus prédictives :")
    for _, row in importance.head(3).iterrows():
        log.info(f"      - {row['feature']} : {row['importance']:.3f}")

    return df


# ============================================================
# 6. Sauvegarde finale
# ============================================================

def save_usage(df: pd.DataFrame) -> Path:
    """Sauvegarde le DataFrame final dans la couche USAGE."""
    USAGE_DIR.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = USAGE_DIR / f"{today_str}.parquet"

    df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

    size_kb = output_path.stat().st_size / 1024
    log.info(f"💾 Sauvegardé : {output_path}")
    log.info(f"   📊 {df.shape[0]} lignes × {df.shape[1]} colonnes — {size_kb:.1f} Ko")
    return output_path


# ============================================================
# Point d'entrée principal
# ============================================================

def run():
    """Fonction principale appelée par le DAG Airflow."""
    log.info("=" * 60)
    log.info("🚀 COMBINATION + ML — Formatted → Usage")
    log.info("=" * 60)

    df = load_and_join()
    df = mark_grails(df)
    df = compute_hype_score(df)
    df = apply_kmeans(df)
    df = apply_random_forest(df)
    output_path = save_usage(df)

    log.info("=" * 60)
    log.info("✅ COMBINATION + ML TERMINÉ — Couche USAGE prête")
    log.info("=" * 60)

    return str(output_path)


# ============================================================
# Standalone (test local hors Docker)
# ============================================================
if __name__ == "__main__":
    IN_DOCKER = Path("/opt/airflow").exists()

    if not IN_DOCKER:
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
        FORMATTED_DIR = PROJECT_ROOT / "tmp" / "data" / "formatted" / "sneakers"
        USAGE_DIR = PROJECT_ROOT / "tmp" / "data" / "usage" / "sneakers_hype"

    run()