"""
Modele de prediction base sur une regression de Poisson (approche de Maher,
1982 -- la base de la plupart des modeles de prediction foot serieux).

Principe :
- Chaque equipe a une "force d'attaque" et une "force de defense".
- On estime ces forces via une regression de Poisson sur l'historique :
  nombre de buts marques ~ force d'attaque de l'equipe qui tire
                          + force de defense de l'equipe qui encaisse
                          + avantage du terrain
- Pour un match a venir, on combine les forces des deux equipes pour obtenir
  le nombre de buts attendus (lambda) de chaque cote.
- On convertit ces lambda en probabilites de victoire/nul/defaite via la
  loi de Poisson (matrice des scores possibles).

C'est un modele simple, transparent (on peut inspecter les forces de chaque
equipe) et bien plus fiable qu'un modele complexe sur un historique limite.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import BASE_DIR, MIN_MATCHES_FOR_PREDICTION

MODEL_PATH = BASE_DIR / "data" / "model.pkl"
MAX_GOALS = 8  # plafond pour la matrice de scores (au-dela, probabilite negligeable)


def _build_training_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme l'historique (1 ligne = 1 match) en format long
    (1 ligne = 1 performance offensive d'une equipe) attendu par la regression.
    """
    home_rows = pd.DataFrame({
        "team": matches["home_team"],
        "opponent": matches["away_team"],
        "goals": matches["home_goals"],
        "is_home": 1,
    })
    away_rows = pd.DataFrame({
        "team": matches["away_team"],
        "opponent": matches["home_team"],
        "goals": matches["away_goals"],
        "is_home": 0,
    })
    return pd.concat([home_rows, away_rows], ignore_index=True)


class PoissonMatchModel:
    def __init__(self):
        self.result_ = None
        self.known_teams_ = set()
        self.league_ = None

    def fit(self, matches: pd.DataFrame, league: str):
        """Entraine le modele sur l'historique d'une ligue donnee."""
        if matches.empty:
            raise ValueError(f"Aucune donnee d'entrainement pour la ligue '{league}'.")

        long_df = _build_training_frame(matches)

        # filtre les equipes avec trop peu de matchs (debut de dataset instable)
        counts = long_df["team"].value_counts()
        valid_teams = counts[counts >= MIN_MATCHES_FOR_PREDICTION].index
        long_df = long_df[long_df["team"].isin(valid_teams) & long_df["opponent"].isin(valid_teams)]

        if long_df.empty:
            raise ValueError(f"Pas assez de matchs pour entrainer un modele fiable sur '{league}'.")

        model = smf.glm(
            formula="goals ~ is_home + C(team) + C(opponent)",
            data=long_df,
            family=sm.families.Poisson(),
        )
        self.result_ = model.fit()
        self.known_teams_ = set(valid_teams)
        self.league_ = league
        return self

    def _expected_goals(self, team: str, opponent: str, is_home: bool) -> float:
        row = pd.DataFrame({
            "team": [team],
            "opponent": [opponent],
            "is_home": [1 if is_home else 0],
        })
        return float(self.result_.predict(row).iloc[0])

    def predict_match(self, home_team: str, away_team: str) -> dict:
        """
        Renvoie les buts attendus et les probabilites 1X2 pour un match.
        Leve une ValueError si une des deux equipes est inconnue du modele
        (pas assez d'historique) -- mieux vaut le signaler que d'inventer un chiffre.
        """
        if home_team not in self.known_teams_:
            raise ValueError(f"Equipe inconnue du modele : '{home_team}'")
        if away_team not in self.known_teams_:
            raise ValueError(f"Equipe inconnue du modele : '{away_team}'")

        lambda_home = self._expected_goals(home_team, away_team, is_home=True)
        lambda_away = self._expected_goals(away_team, home_team, is_home=False)

        # matrice des probabilites de chaque score exact possible
        home_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_home)
        away_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_away)
        score_matrix = np.outer(home_probs, away_probs)

        prob_home = float(np.tril(score_matrix, -1).sum())   # home_goals > away_goals
        prob_draw = float(np.trace(score_matrix))
        prob_away = float(np.triu(score_matrix, 1).sum())    # away_goals > home_goals

        # normalisation (la troncature a MAX_GOALS laisse un tout petit residu)
        total = prob_home + prob_draw + prob_away
        prob_home, prob_draw, prob_away = prob_home / total, prob_draw / total, prob_away / total

        return {
            "lambda_home": round(lambda_home, 3),
            "lambda_away": round(lambda_away, 3),
            "prob_home": round(prob_home, 4),
            "prob_draw": round(prob_draw, 4),
            "prob_away": round(prob_away, 4),
        }

    def save(self, path: Path = None):
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = None) -> "PoissonMatchModel":
        path = path or MODEL_PATH
        with open(path, "rb") as f:
            return pickle.load(f)


def train_all_leagues(leagues_matches: dict) -> dict:
    """
    Entraine un modele separe par ligue (les forces d'attaque/defense n'ont
    de sens qu'a l'interieur d'un meme championnat).
    leagues_matches : dict {league_key: DataFrame des matchs}
    Renvoie : dict {league_key: PoissonMatchModel}
    """
    models = {}
    for league_key, matches in leagues_matches.items():
        try:
            model = PoissonMatchModel().fit(matches, league_key)
            models[league_key] = model
            print(f"  {league_key} : modele entraine sur {len(matches)} matchs, "
                  f"{len(model.known_teams_)} equipes reconnues")
        except ValueError as exc:
            print(f"  [!] {league_key} : {exc}")
    return models
