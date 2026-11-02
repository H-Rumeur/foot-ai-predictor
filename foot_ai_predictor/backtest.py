"""
Backtesting : est-ce que le modele est vraiment fiable ?

Methode : validation "walk-forward". On entraine le modele uniquement sur les
matchs passes par rapport a une date donnee, puis on l'evalue sur les matchs
qui suivent -- jamais sur des matchs qu'il a deja "vus". C'est la seule facon
honnete de tester un modele de prediction (sinon on triche en lui laissant
voir le futur).

Deux angles d'evaluation :
1. Metriques de qualite : log loss et score de Brier (plus bas = meilleur).
   Un modele qui prediraait 33/33/33 partout a un score de reference a battre.
2. Calibration : quand le modele dit "60% de victoire domicile", est-ce que
   ca arrive vraiment ~60% du temps sur un grand nombre de matchs ? Un modele
   mal calibre (ex: annonce 70% mais gagne 50% du temps) est dangereux a
   utiliser pour miser, meme s'il "devine" souvent le bon resultat.

Usage : python pipeline.py backtest
"""
import numpy as np
import pandas as pd

from model import PoissonMatchModel

RESULT_TO_INDEX = {"H": 0, "D": 1, "A": 2}


def _log_loss_and_brier(rows: list[dict]) -> dict:
    """Calcule log loss et score de Brier a partir des predictions/resultats stockes."""
    eps = 1e-15
    log_losses = []
    briers = []
    for r in rows:
        probs = [r["prob_home"], r["prob_draw"], r["prob_away"]]
        actual_idx = RESULT_TO_INDEX[r["actual_result"]]
        p_actual = max(min(probs[actual_idx], 1 - eps), eps)
        log_losses.append(-np.log(p_actual))

        actual_vec = [1 if i == actual_idx else 0 for i in range(3)]
        briers.append(sum((p - a) ** 2 for p, a in zip(probs, actual_vec)))

    return {
        "log_loss": float(np.mean(log_losses)),
        "brier_score": float(np.mean(briers)),
        "n_matches": len(rows),
    }


def _calibration_table(rows: list[dict], n_bins: int = 5) -> pd.DataFrame:
    """
    Regroupe les predictions par tranche de probabilite (ex: 40-60%) et compare
    la probabilite moyenne annoncee a la frequence reelle observee.
    """
    records = []
    for r in rows:
        records.append({"prob": r["prob_home"], "hit": 1 if r["actual_result"] == "H" else 0})
        records.append({"prob": r["prob_draw"], "hit": 1 if r["actual_result"] == "D" else 0})
        records.append({"prob": r["prob_away"], "hit": 1 if r["actual_result"] == "A" else 0})

    df = pd.DataFrame(records)
    df["bin"] = pd.cut(df["prob"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)

    summary = df.groupby("bin", observed=True).agg(
        probabilite_moyenne_predite=("prob", "mean"),
        frequence_reelle_observee=("hit", "mean"),
        nb_observations=("hit", "count"),
    ).reset_index()
    summary["bin"] = summary["bin"].astype(str)  # Altair/Streamlit n'accepte pas les Interval pandas comme labels
    return summary


def run_backtest(matches: pd.DataFrame, league: str, min_train_matches: int = 150,
                  step: int = 30) -> dict:
    """
    Entraine/evalue le modele de facon progressive sur l'historique d'une ligue.

    matches : DataFrame trie par date, toutes les saisons confondues.
    min_train_matches : nombre minimum de matchs avant le premier entrainement.
    step : tous les combien de matchs on ré-entraine le modele (pour aller plus
           vite -- ré-entrainer a chaque match serait tres lent et peu different).
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    if len(matches) < min_train_matches + step:
        return {"error": f"Pas assez de matchs ({len(matches)}) pour backtester '{league}'."}

    prediction_rows = []
    cursor = min_train_matches

    while cursor < len(matches):
        train_df = matches.iloc[:cursor]
        test_df = matches.iloc[cursor: cursor + step]

        try:
            model = PoissonMatchModel().fit(train_df, league)
        except ValueError:
            cursor += step
            continue

        for _, row in test_df.iterrows():
            try:
                pred = model.predict_match(row["home_team"], row["away_team"])
            except ValueError:
                continue  # equipe promue/pas assez d'historique -> on saute
            prediction_rows.append({
                **pred,
                "actual_result": row["result"],
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            })

        cursor += step

    if not prediction_rows:
        return {"error": f"Aucune prediction backtestee generee pour '{league}'."}

    metrics = _log_loss_and_brier(prediction_rows)
    calibration = _calibration_table(prediction_rows)

    # reference : un modele "naif" qui prediraait toujours les frequences globales
    baseline_probs = matches["result"].value_counts(normalize=True)
    baseline_rows = [{
        "prob_home": baseline_probs.get("H", 0),
        "prob_draw": baseline_probs.get("D", 0),
        "prob_away": baseline_probs.get("A", 0),
        "actual_result": r["actual_result"],
    } for r in prediction_rows]
    baseline_metrics = _log_loss_and_brier(baseline_rows)

    return {
        "league": league,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "calibration": calibration,
        "predictions": pd.DataFrame(prediction_rows),
    }


def print_backtest_report(result: dict):
    if "error" in result:
        print(f"[!] {result['error']}")
        return

    m, b = result["metrics"], result["baseline_metrics"]
    print(f"\n=== Backtest : {result['league']} ({m['n_matches']} matchs testes) ===")
    print(f"Log loss du modele    : {m['log_loss']:.4f}  (reference naive : {b['log_loss']:.4f})")
    print(f"Score de Brier modele : {m['brier_score']:.4f}  (reference naive : {b['brier_score']:.4f})")
    if m["log_loss"] < b["log_loss"]:
        print("-> Le modele bat la reference naive. Bon signe.")
    else:
        print("-> Le modele ne bat PAS la reference naive. A ne pas utiliser tel quel.")

    print("\nCalibration (probabilite predite vs frequence reelle observee) :")
    print(result["calibration"].to_string(index=False))
