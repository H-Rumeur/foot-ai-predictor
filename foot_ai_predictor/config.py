"""
Configuration centrale du projet.
Modifie les valeurs ici pour changer les ligues suivies, les saisons
historiques a telecharger, ou l'emplacement de la base de donnees.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # charge les variables depuis le fichier .env s'il existe

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "football.db"

# Cle API football-data.org (gratuite) -> https://www.football-data.org/client/register
FOOTBALL_DATA_ORG_API_KEY = os.getenv("FOOTBALL_DATA_ORG_API_KEY", "")
FOOTBALL_DATA_ORG_BASE_URL = "https://api.football-data.org/v4"

# Les 6 ligues suivies : nom affiche, code football-data.org (API live),
# code football-data.co.uk (CSV historique), et pays (pour l'affichage).
LEAGUES = {
    "premier_league": {
        "display_name": "Premier League (Angleterre)",
        "fdorg_code": "PL",
        "fdcouk_code": "E0",
    },
    "la_liga": {
        "display_name": "La Liga (Espagne)",
        "fdorg_code": "PD",
        "fdcouk_code": "SP1",
    },
    "bundesliga": {
        "display_name": "Bundesliga (Allemagne)",
        "fdorg_code": "BL1",
        "fdcouk_code": "D1",
    },
    "serie_a": {
        "display_name": "Serie A (Italie)",
        "fdorg_code": "SA",
        "fdcouk_code": "I1",
    },
    "ligue_1": {
        "display_name": "Ligue 1 (France)",
        "fdorg_code": "FL1",
        "fdcouk_code": "F1",
    },
    "eredivisie": {
        "display_name": "Eredivisie (Pays-Bas)",
        "fdorg_code": "DED",
        "fdcouk_code": "N1",
    },
}

# Saisons historiques a telecharger pour l'entrainement / le backtesting.
# Format attendu par football-data.co.uk : "2324" pour la saison 2023-2024.
# Ajuste cette liste si tu veux plus ou moins d'historique.
HISTORICAL_SEASONS = ["2122", "2223", "2324", "2425", "2526"]

FOOTBALL_DATA_CO_UK_BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Nombre minimum de matchs joues par une equipe (domicile + exterieur confondus)
# avant que le modele accepte de produire une prediction fiable pour elle.
MIN_MATCHES_FOR_PREDICTION = 5

# Fraction de Kelly appliquee par defaut (Kelly complet = 1.0).
# 0.25-0.5 est recommande pour limiter la variance.
DEFAULT_KELLY_FRACTION = 0.3
