"""
Recupere les matchs a venir (et les resultats recents de la saison en cours)
via l'API gratuite football-data.org.

Necessite une cle API gratuite : https://www.football-data.org/client/register
A placer dans le fichier .env sous FOOTBALL_DATA_ORG_API_KEY=ta_cle

Le plan gratuit est limite a 10 requetes/minute -> ce module respecte
automatiquement cette limite.

Usage : python pipeline.py fetch-live
"""
import sys
import time

import requests

from config import LEAGUES, FOOTBALL_DATA_ORG_API_KEY, FOOTBALL_DATA_ORG_BASE_URL
from db import get_connection, insert_fixture

REQUEST_DELAY_SECONDS = 6.5  # reste sous la limite de 10 requetes/minute


def _headers():
    return {"X-Auth-Token": FOOTBALL_DATA_ORG_API_KEY}


def fetch_upcoming_for_league(league_key: str, days_ahead: int = 14):
    """Recupere les prochains matchs programmes pour une ligue donnee."""
    if not FOOTBALL_DATA_ORG_API_KEY:
        print(
            "[!] Pas de cle API configuree. Ajoute FOOTBALL_DATA_ORG_API_KEY "
            "dans le fichier .env (voir README pour l'obtenir gratuitement)."
        )
        return []

    comp_code = LEAGUES[league_key]["fdorg_code"]
    url = f"{FOOTBALL_DATA_ORG_BASE_URL}/competitions/{comp_code}/matches"
    params = {"status": "SCHEDULED"}

    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=15)
    except requests.RequestException as exc:
        print(f"  [!] Erreur reseau pour {league_key} : {exc}")
        return []

    if resp.status_code == 429:
        print("  [!] Limite de requetes atteinte, attente 60s...")
        time.sleep(60)
        return fetch_upcoming_for_league(league_key, days_ahead)

    if resp.status_code != 200:
        print(f"  [!] {league_key} : reponse HTTP {resp.status_code} - {resp.text[:200]}")
        return []

    data = resp.json()
    matches = data.get("matches", [])
    return matches


def import_all_upcoming(days_ahead: int = 14):
    """Recupere et stocke les matchs a venir pour toutes les ligues configurees."""
    total = 0
    with get_connection() as conn:
        for i, league_key in enumerate(LEAGUES):
            print(f"Ligue : {LEAGUES[league_key]['display_name']}")
            matches = fetch_upcoming_for_league(league_key, days_ahead)
            count = 0
            for m in matches:
                try:
                    insert_fixture(
                        conn,
                        fdorg_id=m["id"],
                        league=league_key,
                        date=m["utcDate"][:10],
                        home_team=m["homeTeam"]["name"],
                        away_team=m["awayTeam"]["name"],
                        status=m["status"],
                    )
                    count += 1
                except (KeyError, TypeError):
                    continue
            conn.commit()
            print(f"  {count} matchs a venir importes")
            total += count

            # respecte la limite de requetes gratuite entre chaque ligue
            if i < len(LEAGUES) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nTermine. {total} matchs a venir au total.")
    return total


if __name__ == "__main__":
    import_all_upcoming()
