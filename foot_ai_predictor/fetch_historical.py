"""
Telecharge l'historique des resultats (plusieurs saisons) depuis les fichiers
CSV gratuits de football-data.co.uk, et les stocke dans la base SQLite.
C'est cette donnee qui sert a entrainer le modele et a le backtester.

Usage : python pipeline.py fetch-historical
"""
import io
import sys

import pandas as pd
import requests

from config import LEAGUES, HISTORICAL_SEASONS, FOOTBALL_DATA_CO_UK_BASE_URL
from db import get_connection, insert_match

# Colonnes utiles dans les CSV football-data.co.uk. Certaines saisons/ligues
# n'ont pas toutes les colonnes de cotes -> on gere ca avec .get().
REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLUMNS = ["B365H", "B365D", "B365A"]


def _parse_date(raw_date: str) -> str:
    """Les CSV utilisent souvent DD/MM/YY ou DD/MM/YYYY selon la saison."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(raw_date, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # dernier recours : laisser pandas deviner
    return pd.to_datetime(raw_date, dayfirst=True).strftime("%Y-%m-%d")


def fetch_one_season(league_key: str, season_code: str) -> pd.DataFrame | None:
    """Telecharge un CSV (une ligue, une saison). Renvoie None si indisponible."""
    div_code = LEAGUES[league_key]["fdcouk_code"]
    url = f"{FOOTBALL_DATA_CO_UK_BASE_URL}/{season_code}/{div_code}.csv"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [!] {league_key} {season_code} indisponible ({exc})")
        return None

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        print(f"  [!] {league_key} {season_code} : erreur de lecture CSV ({exc})")
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [!] {league_key} {season_code} : colonnes manquantes {missing}, ignore")
        return None

    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df


def import_all_history():
    """Telecharge et importe toutes les ligues x toutes les saisons configurees."""
    total_inserted = 0
    with get_connection() as conn:
        for league_key in LEAGUES:
            print(f"Ligue : {LEAGUES[league_key]['display_name']}")
            for season_code in HISTORICAL_SEASONS:
                df = fetch_one_season(league_key, season_code)
                if df is None:
                    continue

                count = 0
                for _, row in df.iterrows():
                    try:
                        date_str = _parse_date(row["Date"])
                        insert_match(
                            conn,
                            league=league_key,
                            season=season_code,
                            date=date_str,
                            home_team=str(row["HomeTeam"]).strip(),
                            away_team=str(row["AwayTeam"]).strip(),
                            home_goals=int(row["FTHG"]),
                            away_goals=int(row["FTAG"]),
                            result=str(row["FTR"]).strip(),
                            b365h=row.get("B365H") if "B365H" in df.columns else None,
                            b365d=row.get("B365D") if "B365D" in df.columns else None,
                            b365a=row.get("B365A") if "B365A" in df.columns else None,
                        )
                        count += 1
                    except (ValueError, TypeError) as exc:
                        continue  # ligne malformee, on saute
                conn.commit()
                print(f"  {season_code} : {count} matchs importes")
                total_inserted += count

    print(f"\nTermine. {total_inserted} lignes traitees au total.")
    if total_inserted == 0:
        print(
            "Aucune donnee importee. Verifie ta connexion internet, ou que "
            "l'URL de football-data.co.uk n'a pas change (voir README)."
        )


if __name__ == "__main__":
    import_all_history()
