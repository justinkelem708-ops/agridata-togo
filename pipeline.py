"""
AgriData Togo — Orchestrateur principal
Tu contrôles le nœud. Le pipeline s'exécute seul.

Architecture :
  [Collecte terrain] → [Enrichissement] → [Modèle] → [Dashboard] → [Alertes]

Usage : python pipeline.py
"""

import os
import sys
import time
import logging
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agridata_pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AgriData")

# ─── Chemins ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
AGENTS_DIR = BASE_DIR / "Agents"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ─── Agent 1 : Collecteur de données (WFP + terrain) ─────────────────────────
class CollectorAgent:
    """
    Responsable : télécharger et agréger toutes les sources de données.
    Sources : WFP/HDX (historique) + Google Forms (terrain en temps réel)
    """
    WFP_URL = (
        "https://data.humdata.org/dataset/wfp-food-prices-for-togo"
        "/resource/6740fe96-c884-4165-b6f1-a21bd349be0e/download"
        "/wfp_food_prices_togo.csv"
    )
    GFORMS_CSV = None  # ← colle ici l'URL CSV d'export de ton Google Forms

    def run(self):
        log.info("=== Agent 1 : Collecteur ===")
        results = {}

        # WFP historique
        wfp_path = RAW_DIR / "wfp_food_prices_togo.csv"
        if not wfp_path.exists():
            log.info("Téléchargement WFP/HDX...")
            try:
                import requests
                r = requests.get(self.WFP_URL, timeout=60)
                r.raise_for_status()
                wfp_path.write_bytes(r.content)
                log.info(f"WFP téléchargé : {len(r.content)//1024} Ko")
            except Exception as e:
                log.warning(f"WFP téléchargement échoué : {e}")
        else:
            log.info(f"WFP déjà présent : {wfp_path}")
        results["wfp"] = wfp_path

        # Données terrain (Google Forms → CSV)
        terrain_path = RAW_DIR / "terrain_data.csv"
        if self.GFORMS_CSV:
            try:
                import requests
                r = requests.get(self.GFORMS_CSV, timeout=30)
                terrain_path.write_bytes(r.content)
                df_t = pd.read_csv(terrain_path)
                log.info(f"Terrain : {len(df_t)} observations collectées")
            except Exception as e:
                log.warning(f"Terrain non disponible : {e}")
        else:
            log.info("Terrain : Google Forms URL non configurée (voir GFORMS_CSV)")

        return results


# ─── Agent 2 : Enrichisseur (ton enrich_data.py étendu) ──────────────────────
class EnricherAgent:
    """
    Responsable : ajouter météo, taux de change, spread, score d'instabilité.
    Étend ton enrich_data.py existant à TOUTES les villes du Togo.
    """

    CITIES = {
        "Lomé"      : "Lome,TG",
        "Kara"      : "Kara,TG",
        "Atakpamé"  : "Atakpame,TG",
        "Sokodé"    : "Sokode,TG",
        "Dapaong"   : "Dapaong,TG",
    }

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(AGENTS_DIR / ".env")
        self.api_key = os.getenv("OPENWEATHER_API_KEY")

    def get_weather(self, city_code):
        import requests
        try:
            r = requests.get(
                "http://api.openweathermap.org/data/2.5/weather",
                params={"q": city_code, "appid": self.api_key, "units": "metric"},
                timeout=10,
            )
            d = r.json()
            return {
                "temperature"       : d["main"]["temp"],
                "humidity"          : d["main"]["humidity"],
                "weather_condition" : d["weather"][0]["description"],
                "wind_speed"        : d["wind"]["speed"],
                "rain_3h"           : d.get("rain", {}).get("3h", 0),
            }
        except Exception as e:
            log.warning(f"Météo {city_code} : {e}")
            return {k: None for k in ["temperature","humidity","weather_condition","wind_speed","rain_3h"]}

    def get_exchange_rate(self):
        import requests
        try:
            r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
            return r.json()["rates"]["XOF"]
        except Exception as e:
            log.warning(f"Taux de change : {e}")
            return 615.0  # fallback approximatif

    def run(self, df):
        log.info("=== Agent 2 : Enrichisseur ===")

        # Météo par marché (pas juste Lomé)
        weather_cache = {}
        for city_name, city_code in self.CITIES.items():
            weather_cache[city_name] = self.get_weather(city_code)
            time.sleep(0.5)

        def apply_weather(row):
            market = row.get("market", "Lomé")
            city   = next((c for c in self.CITIES if c.lower() in str(market).lower()), "Lomé")
            return pd.Series(weather_cache.get(city, weather_cache["Lomé"]))

        weather_df = df.apply(apply_weather, axis=1)
        df = pd.concat([df, weather_df], axis=1)

        # Taux de change
        rate = self.get_exchange_rate()
        df["usd_xof_rate"] = rate
        if "price" in df.columns:
            df["price_usd"] = (df["price"] / rate).round(4)

        # Spread si données terrain présentes (prix_vendeur / prix_acheteur)
        if "prix_vendeur_fcfa" in df.columns and "prix_acheteur_fcfa" in df.columns:
            df["price_spread_fcfa"] = df["prix_vendeur_fcfa"] - df["prix_acheteur_fcfa"]
            df["price_spread_pct"]  = (df["price_spread_fcfa"] / df["prix_vendeur_fcfa"] * 100).round(2)
            df["price_instability_score"] = pd.cut(
                df["price_spread_pct"].fillna(0),
                bins=[-1, 5, 15, 100], labels=[1, 2, 3]
            )
            log.info(f"Spread calculé : moy {df['price_spread_pct'].mean():.1f}%")

        # Saison agricole (variable critique pour la prédiction)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["saison"] = df["date"].dt.month.map({
                1:"sèche", 2:"sèche", 3:"pre_pluies",
                4:"pluies_N", 5:"pluies_N", 6:"pluies_N",
                7:"pluies_S", 8:"pluies_S", 9:"pluies_S",
                10:"récolte", 11:"récolte", 12:"sèche"
            })

        df["enriched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        log.info(f"Enrichissement terminé : {len(df)} lignes, {len(df.columns)} colonnes")
        return df


# ─── Agent 3 : Modèle de prédiction ──────────────────────────────────────────
class ModelAgent:
    """
    Responsable : entraîner Prophet sur chaque paire (produit × marché)
    et produire les prédictions à 30/60/90 jours.
    """

    PRODUCTS_PRIORITY = [
        "Maize", "Maize (white)", "Millet", "Sorghum",
        "Cassava", "Rice", "Cowpeas", "Oil (palm)",
        "Yams", "Groundnuts (shelled)",
    ]

    def run(self, df, horizon_months=3):
        log.info("=== Agent 3 : Modèle ===")
        from prophet import Prophet
        predictions = []

        commodities = [c for c in self.PRODUCTS_PRIORITY if c in df["commodity"].unique()]
        commodities += [c for c in df["commodity"].unique() if c not in commodities]

        markets = df["market"].unique() if "market" in df.columns else ["National"]

        for commodity in commodities:
            for market in markets:
                mask = (df["commodity"] == commodity)
                if "market" in df.columns:
                    mask &= (df["market"] == market)

                sub = df[mask][["date", "price"]].dropna().sort_values("date")
                sub.columns = ["ds", "y"]

                if len(sub) < 24:
                    continue

                try:
                    m = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode="multiplicative",
                        changepoint_prior_scale=0.05,
                    )
                    m.fit(sub)
                    future  = m.make_future_dataframe(periods=horizon_months, freq="MS")
                    fc      = m.predict(future)
                    fc_tail = fc.tail(horizon_months)

                    for _, row in fc_tail.iterrows():
                        predictions.append({
                            "commodity"   : commodity,
                            "market"      : market,
                            "date_pred"   : row["ds"].strftime("%Y-%m"),
                            "price_pred"  : round(row["yhat"], 0),
                            "price_low"   : round(row["yhat_lower"], 0),
                            "price_high"  : round(row["yhat_upper"], 0),
                            "predicted_at": datetime.now().strftime("%Y-%m-%d"),
                        })

                except Exception as e:
                    log.debug(f"Modèle {commodity}/{market} : {e}")
                    continue

        df_pred = pd.DataFrame(predictions)
        out     = PROC_DIR / "predictions.csv"
        df_pred.to_csv(out, index=False)
        log.info(f"Prédictions sauvegardées : {len(df_pred)} lignes → {out}")
        return df_pred


# ─── Agent 4 : Alertes ───────────────────────────────────────────────────────
class AlertAgent:
    """
    Responsable : détecter les anomalies de prix et envoyer des alertes.
    Seuils : prix prédit > moy+30% → ALERTE ROUGE
             prix prédit > moy+15% → ALERTE ORANGE
    """

    def run(self, df_prices, df_predictions):
        log.info("=== Agent 4 : Alertes ===")
        alerts = []

        avg_prices = df_prices.groupby("commodity")["price"].mean()

        for _, row in df_predictions.iterrows():
            commodity = row["commodity"]
            pred      = row["price_pred"]
            avg       = avg_prices.get(commodity)

            if avg is None or avg == 0:
                continue

            deviation = (pred - avg) / avg * 100

            if deviation > 30:
                level = "ROUGE"
            elif deviation > 15:
                level = "ORANGE"
            else:
                continue

            alert = {
                "niveau"    : level,
                "commodity" : commodity,
                "market"    : row["market"],
                "date"      : row["date_pred"],
                "prix_prédit" : pred,
                "prix_moyen"  : round(avg, 0),
                "écart_%"     : round(deviation, 1),
                "message"   : (
                    f"⚠️ ALERTE {level} — {commodity} à {row['market']} : "
                    f"prix prédit {pred:,.0f} FCFA ({deviation:+.0f}% vs moy {avg:,.0f} FCFA)"
                    f" en {row['date_pred']}"
                ),
            }
            alerts.append(alert)
            log.warning(alert["message"])

        df_alerts = pd.DataFrame(alerts)
        if len(df_alerts) > 0:
            out = PROC_DIR / "alerts.csv"
            df_alerts.to_csv(out, index=False)
            log.info(f"{len(df_alerts)} alertes sauvegardées → {out}")
        else:
            log.info("Aucune alerte détectée.")

        return df_alerts


# ─── Orchestrateur ────────────────────────────────────────────────────────────
class Pipeline:
    """
    Tu es le nœud de contrôle. Le pipeline s'exécute automatiquement.
    Lance : python pipeline.py
    """

    def __init__(self):
        self.collector  = CollectorAgent()
        self.enricher   = EnricherAgent()
        self.model      = ModelAgent()
        self.alerter    = AlertAgent()

    def run(self, horizon_months=3):
        start = datetime.now()
        log.info("=" * 55)
        log.info("  AgriData Togo — Pipeline complet")
        log.info(f"  Démarrage : {start.strftime('%Y-%m-%d %H:%M')}")
        log.info("=" * 55)

        # Étape 1 — Collecte
        sources = self.collector.run()

        # Étape 2 — Chargement et fusion des données
        dfs = []
        wfp_path = sources.get("wfp")
        if wfp_path and Path(wfp_path).exists():
            df_wfp = pd.read_csv(wfp_path, low_memory=False)
            # Nettoyage minimal WFP
            if df_wfp.iloc[0, 0] in ["date", "Date"]:
                df_wfp = df_wfp.iloc[1:].reset_index(drop=True)
            df_wfp.columns = df_wfp.columns.str.lower().str.strip()
            col_map = {"city": "market", "admin1": "region"}
            df_wfp = df_wfp.rename(columns=col_map)
            for fmt in ["%Y-%m-%d", "%m/%d/%Y"]:
                try:
                    df_wfp["date"] = pd.to_datetime(df_wfp["date"], format=fmt)
                    break
                except Exception:
                    continue
            df_wfp["price"] = pd.to_numeric(df_wfp["price"], errors="coerce")
            df_wfp = df_wfp.dropna(subset=["price", "date"])
            df_wfp = df_wfp[df_wfp["price"] > 0]
            dfs.append(df_wfp)
            log.info(f"WFP chargé : {len(df_wfp)} lignes")

        terrain_path = RAW_DIR / "terrain_data.csv"
        if terrain_path.exists():
            df_terrain = pd.read_csv(terrain_path)
            dfs.append(df_terrain)
            log.info(f"Terrain chargé : {len(df_terrain)} lignes")

        if not dfs:
            log.error("Aucune donnée disponible. Relance download_data.py.")
            return

        df = pd.concat(dfs, ignore_index=True)

        # Étape 3 — Enrichissement
        df_enriched = self.enricher.run(df)
        out_enriched = PROC_DIR / "enriched_data.csv"
        df_enriched.to_csv(out_enriched, index=False)

        # Étape 4 — Modèle
        df_pred = self.model.run(df_enriched, horizon_months=horizon_months)

        # Étape 5 — Alertes
        df_alerts = self.alerter.run(df_enriched, df_pred)

        # Résumé
        elapsed = (datetime.now() - start).seconds
        log.info("=" * 55)
        log.info(f"  Pipeline terminé en {elapsed}s")
        log.info(f"  Données enrichies : {len(df_enriched)} lignes")
        log.info(f"  Prédictions : {len(df_pred)} entrées")
        log.info(f"  Alertes : {len(df_alerts)}")
        log.info(f"  Lance le dashboard : streamlit run app.py")
        log.info("=" * 55)

        return {
            "enriched"   : df_enriched,
            "predictions": df_pred,
            "alerts"     : df_alerts,
        }


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run(horizon_months=3)
