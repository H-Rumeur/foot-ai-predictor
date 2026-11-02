"""
Rafraichissement des modeles et des predictions. Centralise ici pour que le
CLI (pipeline.py) et le bouton "Actualiser" du dashboard (app.py) partagent
exactement la meme logique, sans dupliquer de code.
"""
import pickle

from config import LEAGUES
from db import get_connection, get_matches_df, get_upcoming_fixtures_df, save_prediction
from model import PoissonMatchModel, MODEL_PATH
from team_mapping import to_historical_name


def retrain_and_predict() -> dict:
    """
    Entraine un modele par ligue sur l'historique deja present en base, sauvegarde
    les modeles, et regenere les predictions pour tous les matchs a venir deja
    en base. Ne telecharge rien (voir fetch_historical.py / fetch_live.py pour ca).
    """
    all_models = {}
    trained_leagues, skipped_leagues = [], []

    for league_key in LEAGUES:
        matches = get_matches_df(league_key)
        if matches.empty:
            skipped_leagues.append(league_key)
            continue
        try:
            all_models[league_key] = PoissonMatchModel().fit(matches, league_key)
            trained_leagues.append(league_key)
        except ValueError:
            skipped_leagues.append(league_key)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(all_models, f)

    n_predictions = 0
    unknown_matches = []
    with get_connection() as conn:
        for league_key, model in all_models.items():
            fixtures = get_upcoming_fixtures_df(league_key)
            for _, fx in fixtures.iterrows():
                home_hist = to_historical_name(fx["home_team"])
                away_hist = to_historical_name(fx["away_team"])
                try:
                    pred = model.predict_match(home_hist, away_hist)
                except ValueError:
                    unknown_matches.append(f"{fx['home_team']} vs {fx['away_team']}")
                    continue
                save_prediction(
                    conn, fixture_id=int(fx["id"]), league=league_key, date=fx["date"],
                    home_team=fx["home_team"], away_team=fx["away_team"],
                    lambda_home=pred["lambda_home"], lambda_away=pred["lambda_away"],
                    prob_home=pred["prob_home"], prob_draw=pred["prob_draw"], prob_away=pred["prob_away"],
                )
                n_predictions += 1
        conn.commit()

    return {
        "trained_leagues": trained_leagues,
        "skipped_leagues": skipped_leagues,
        "n_predictions": n_predictions,
        "unknown_matches": unknown_matches,
    }
