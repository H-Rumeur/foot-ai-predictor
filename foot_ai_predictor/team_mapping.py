"""
Les deux sources de donnees (football-data.org pour les matchs a venir,
football-data.co.uk pour l'historique) nomment parfois les equipes differemment
("Manchester United FC" vs "Man United"). Ce module fait la correspondance.
"""
import json
from pathlib import Path

MAPPING_FILE = Path(__file__).resolve().parent / "team_mapping.json"


def _load_mapping():
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw.pop("_comment", None)
    return raw


_MAPPING = _load_mapping()


def to_historical_name(fdorg_name: str) -> str:
    """
    Convertit un nom d'equipe football-data.org vers le nom equivalent
    utilise dans l'historique football-data.co.uk.
    Si l'equipe n'est pas dans le mapping, renvoie le nom tel quel
    (ca marchera si les deux sources utilisent deja le meme nom).
    """
    return _MAPPING.get(fdorg_name, fdorg_name)


def unknown_teams(fdorg_names, historical_names):
    """
    Utilitaire de diagnostic : parmi une liste de noms football-data.org,
    lesquels -- une fois convertis -- ne correspondent a aucune equipe
    presente dans l'historique ? Utile pour completer team_mapping.json.
    """
    historical_set = set(historical_names)
    missing = []
    for name in fdorg_names:
        converted = to_historical_name(name)
        if converted not in historical_set:
            missing.append((name, converted))
    return missing
