"""
Dashboard local. Lancer avec :  streamlit run app.py

Quatre onglets :
- Prochains matchs : probabilites du modele pour les matchs a venir
- Performance du modele : resultats du backtest, calibration
- Calculateur de mise : entre une cote, obtiens l'edge et la mise suggeree (Kelly)
- Journal de paris : suivi de tes paris reels places, pour mesurer ta rentabilite
"""
import pickle
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from config import LEAGUES, DEFAULT_KELLY_FRACTION, MIN_MATCHES_FOR_PREDICTION
from db import get_connection, get_matches_df, get_upcoming_fixtures_df, init_db
from model import MODEL_PATH
from backtest import run_backtest
from kelly import evaluate_bet
from refresh import retrain_and_predict
from fetch_live import import_all_upcoming
from fetch_historical import import_all_history

st.set_page_config(page_title="Foot AI Predictor", page_icon="⚽", layout="wide")


# ---------- utilitaires ----------

@st.cache_resource
def load_models():
    if not MODEL_PATH.exists():
        return {}
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def league_options():
    return {v["display_name"]: k for k, v in LEAGUES.items()}


# ---------- verification de premier lancement ----------

if not MODEL_PATH.parent.exists() or not (MODEL_PATH.parent / "football.db").exists():
    init_db()

with st.sidebar:
    st.markdown("### 🔄 Actualisation")
    st.caption("Remplace le passage par le terminal pour garder les predictions a jour.")

    if st.button("Actualiser (matchs a venir + modele)", width="stretch", type="primary"):
        try:
            with st.spinner("Recuperation des matchs a venir (peut prendre ~1 min, limite API gratuite)..."):
                n_fixtures = import_all_upcoming()
            with st.spinner("Re-entrainement du modele et generation des predictions..."):
                result = retrain_and_predict()
            load_models.clear()
            msg = f"{n_fixtures} matchs a venir, {result['n_predictions']} predictions generees."
            if result["skipped_leagues"]:
                msg += f" ({len(result['skipped_leagues'])} ligue(s) sans donnees suffisantes.)"
            st.success(msg)
            st.rerun()
        except Exception as exc:
            st.error(f"Erreur pendant l'actualisation : {exc}")

    with st.expander("Mise a jour complete (historique inclus, ~2-3 min)"):
        st.caption(
            "A faire de temps en temps (ex: une fois par mois) pour que le modele "
            "integre les resultats les plus recents dans son historique d'entrainement."
        )
        if st.button("Tout reactualiser", width="stretch"):
            try:
                with st.spinner("Telechargement de l'historique complet..."):
                    import_all_history()
                with st.spinner("Recuperation des matchs a venir..."):
                    n_fixtures = import_all_upcoming()
                with st.spinner("Re-entrainement du modele..."):
                    result = retrain_and_predict()
                load_models.clear()
                st.success(f"Tout est a jour. {result['n_predictions']} predictions generees.")
                st.rerun()
            except Exception as exc:
                st.error(f"Erreur pendant l'actualisation : {exc}")

models = load_models()

st.title("⚽ Foot AI Predictor")
st.caption(
    "Outil personnel d'aide a la decision base sur un modele statistique. "
    "Ne constitue pas un conseil financier -- les probabilites sont des estimations, pas des certitudes."
)

if not models:
    st.warning(
        "Aucun modele entraine pour l'instant. Lance dans un terminal, a la racine du projet :\n\n"
        "```\npython pipeline.py run-all\n```\n\n"
        "(ou etape par etape : `init-db`, `fetch-historical`, `fetch-live`, `train`, `backtest` -- voir le README)"
    )
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📅 Prochains matchs", "📊 Performance du modele", "🧮 Calculateur de mise", "📒 Journal de paris"]
)

# ---------- onglet 1 : prochains matchs ----------

with tab1:
    st.subheader("Matchs a venir et probabilites du modele")

    league_display = st.selectbox("Ligue", options=["Toutes"] + list(league_options().keys()))
    league_key = None if league_display == "Toutes" else league_options()[league_display]

    with get_connection() as conn:
        query = """
            SELECT p.date, p.league, p.home_team, p.away_team,
                   p.lambda_home, p.lambda_away, p.prob_home, p.prob_draw, p.prob_away
            FROM predictions p
        """
        params = ()
        if league_key:
            query += " WHERE p.league = ?"
            params = (league_key,)
        query += " ORDER BY p.date"
        preds = pd.read_sql_query(query, conn, params=params)

    if preds.empty:
        st.info("Pas de predictions disponibles. Lance `python pipeline.py fetch-live` puis `train`.")
    else:
        preds["league"] = preds["league"].map(lambda k: LEAGUES.get(k, {}).get("display_name", k))
        for col in ["prob_home", "prob_draw", "prob_away"]:
            preds[col] = (preds[col] * 100).round(1)

        display_df = preds.rename(columns={
            "date": "Date", "league": "Ligue", "home_team": "Domicile", "away_team": "Exterieur",
            "lambda_home": "Buts attendus (dom.)", "lambda_away": "Buts attendus (ext.)",
            "prob_home": "% Victoire dom.", "prob_draw": "% Nul", "prob_away": "% Victoire ext.",
        })
        st.dataframe(display_df, width="stretch", hide_index=True)

# ---------- onglet 2 : performance du modele ----------

with tab2:
    st.subheader("Fiabilite du modele (backtesting)")
    st.caption(
        "Le modele est evalue uniquement sur des matchs qu'il n'a jamais vus pendant son entrainement -- "
        "c'est la seule facon honnete de savoir s'il est fiable."
    )

    league_display_bt = st.selectbox(
        "Ligue a evaluer", options=list(league_options().keys()), key="backtest_league"
    )
    league_key_bt = league_options()[league_display_bt]

    if st.button("Lancer le backtest"):
        matches = get_matches_df(league_key_bt)
        if matches.empty:
            st.error("Aucune donnee historique pour cette ligue. Lance `fetch-historical` d'abord.")
        else:
            with st.spinner("Backtest en cours (peut prendre une minute)..."):
                result = run_backtest(matches, league_key_bt)

            if "error" in result:
                st.error(result["error"])
            else:
                m, b = result["metrics"], result["baseline_metrics"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Matchs testes", m["n_matches"])
                col2.metric("Log loss (modele)", f"{m['log_loss']:.4f}",
                            delta=f"{m['log_loss'] - b['log_loss']:.4f} vs reference",
                            delta_color="inverse")
                col3.metric("Score de Brier (modele)", f"{m['brier_score']:.4f}",
                            delta=f"{m['brier_score'] - b['brier_score']:.4f} vs reference",
                            delta_color="inverse")

                if m["log_loss"] < b["log_loss"]:
                    st.success("Le modele bat une reference naive (frequences globales). Bon signe.")
                else:
                    st.error("Le modele ne bat PAS la reference naive -- a ne pas utiliser pour miser en l'etat.")

                st.markdown("**Calibration** — la probabilité prédite doit être proche de la fréquence réelle :")
                calib = result["calibration"].copy()
                calib.columns = ["Tranche de probabilite", "Probabilite predite (moy.)",
                                  "Frequence reelle observee", "Nb observations"]
                st.dataframe(calib, width="stretch", hide_index=True)
                st.line_chart(
                    calib.set_index("Tranche de probabilite")[
                        ["Probabilite predite (moy.)", "Frequence reelle observee"]
                    ]
                )

                st.divider()
                st.markdown("**🔍 Explorer les prédictions passées**")
                st.caption(
                    "Match par match, ce que le modele aurait predit a l'epoque -- sans jamais "
                    "avoir vu le resultat -- et ce qui s'est reellement passe."
                )

                RESULT_LABELS = {"H": "Domicile", "D": "Nul", "A": "Exterieur"}
                explore = result["predictions"].copy()
                explore["resultat_reel"] = explore["actual_result"].map(RESULT_LABELS)
                explore["pronostic_modele"] = explore[["prob_home", "prob_draw", "prob_away"]].idxmax(axis=1).map(
                    {"prob_home": "Domicile", "prob_draw": "Nul", "prob_away": "Exterieur"}
                )
                explore["correct"] = explore["pronostic_modele"] == explore["resultat_reel"]

                accuracy = explore["correct"].mean() * 100
                st.metric(
                    "Taux de bonne prediction (issue la plus probable selon le modele)",
                    f"{accuracy:.1f}%",
                    help="Part des matchs ou l'issue jugee la plus probable par le modele s'est "
                         "realisee. A titre de reference, parier 'domicile' a chaque fois donne "
                         "generalement 45-46% en foot -- un bon signe si le modele fait mieux.",
                )

                fcol1, fcol2 = st.columns([1, 2])
                filter_choice = fcol1.radio(
                    "Filtrer", ["Tous", "Corrects", "Incorrects"], horizontal=True, key="explore_filter"
                )
                team_search = fcol2.text_input("Filtrer par equipe (optionnel)", key="explore_search")

                if filter_choice == "Corrects":
                    explore = explore[explore["correct"]]
                elif filter_choice == "Incorrects":
                    explore = explore[~explore["correct"]]
                if team_search:
                    mask = (
                        explore["home_team"].str.contains(team_search, case=False, na=False)
                        | explore["away_team"].str.contains(team_search, case=False, na=False)
                    )
                    explore = explore[mask]

                explore = explore.sort_values("date", ascending=False).copy()
                for col in ["prob_home", "prob_draw", "prob_away"]:
                    explore[col] = (explore[col] * 100).round(1)
                explore["correct"] = explore["correct"].map({True: "✅", False: "❌"})

                display_explore = explore.rename(columns={
                    "date": "Date", "home_team": "Domicile", "away_team": "Exterieur",
                    "prob_home": "% Dom.", "prob_draw": "% Nul", "prob_away": "% Ext.",
                    "pronostic_modele": "Pronostic du modele", "resultat_reel": "Resultat reel",
                    "correct": "Juste ?",
                })[["Date", "Domicile", "Exterieur", "% Dom.", "% Nul", "% Ext.",
                    "Pronostic du modele", "Resultat reel", "Juste ?"]]

                st.dataframe(display_explore, width="stretch", hide_index=True)
                st.caption(f"{len(display_explore)} match(s) affiche(s) sur {len(result['predictions'])} backtestes au total.")

# ---------- onglet 3 : calculateur de mise ----------

with tab3:
    st.subheader("Calculateur de mise (critere de Kelly)")
    st.caption(
        "Entre la cote proposee par ton bookmaker pour une issue donnee -- l'outil compare "
        "a la probabilite du modele et te dit s'il y a de la valeur, avec une mise suggeree."
    )

    with get_connection() as conn:
        preds_list = pd.read_sql_query(
            "SELECT id, date, league, home_team, away_team, prob_home, prob_draw, prob_away "
            "FROM predictions ORDER BY date", conn
        )

    if preds_list.empty:
        st.info("Aucune prediction disponible pour l'instant.")
    else:
        preds_list["label"] = preds_list.apply(
            lambda r: f"{r['date']} — {r['home_team']} vs {r['away_team']}", axis=1
        )
        chosen_label = st.selectbox("Match", options=preds_list["label"])
        row = preds_list[preds_list["label"] == chosen_label].iloc[0]

        selection = st.radio(
            "Issue sur laquelle tu veux parier",
            options=["Victoire domicile", "Nul", "Victoire exterieur"],
            horizontal=True,
        )
        prob_map = {
            "Victoire domicile": row["prob_home"],
            "Nul": row["prob_draw"],
            "Victoire exterieur": row["prob_away"],
        }
        model_prob = prob_map[selection]
        st.metric("Probabilite du modele pour cette issue", f"{model_prob * 100:.1f}%")

        col1, col2, col3 = st.columns(3)
        odds = col1.number_input("Cote decimale du bookmaker", min_value=1.01, value=2.00, step=0.05)
        bankroll = col2.number_input("Ta bankroll totale (€)", min_value=1.0, value=100.0, step=10.0)
        kelly_frac = col3.slider("Fraction de Kelly appliquee", min_value=0.1, max_value=1.0,
                                  value=DEFAULT_KELLY_FRACTION, step=0.05,
                                  help="25-50% recommande pour limiter la variance liee a l'incertitude du modele")

        st.divider()
        st.markdown("**💰 Combien tu peux gagner**")
        custom_stake = st.number_input(
            "Mise a tester (€) — modifie librement pour voir le gain correspondant",
            min_value=0.0, value=10.0, step=5.0,
        )
        gain_if_win = custom_stake * (odds - 1)
        total_return_if_win = custom_stake * odds
        g1, g2, g3 = st.columns(3)
        g1.metric("Si tu gagnes, tu empoches", f"+{gain_if_win:.2f} €")
        g2.metric("Montant total recupere", f"{total_return_if_win:.2f} €")
        g3.metric("Si tu perds", f"-{custom_stake:.2f} €")

        result = evaluate_bet(model_prob, odds, bankroll, kelly_frac)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilite implicite du bookmaker", f"{result['implied_prob_bookmaker'] * 100:.1f}%")
        c2.metric("Edge (modele - marche)", f"{result['edge'] * 100:+.1f} pts")
        c3.metric("Valeur esperee / unite misee", f"{result['expected_value_per_unit']:+.3f}")

        if result["has_value"]:
            st.success(
                f"Le modele estime une valeur positive sur ce pari. "
                f"Mise suggeree (Kelly {kelly_frac:.0%}) : **{result['suggested_stake']} €**"
            )
        else:
            st.warning(
                "Pas de valeur detectee (le modele est moins optimiste que le bookmaker sur cette issue). "
                "Kelly recommande de ne pas parier."
            )

        if st.button("Enregistrer ce pari dans le journal"):
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO bets (date, league, home_team, away_team, selection,
                                       model_prob, odds, stake, outcome, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (row["date"], row["league"], row["home_team"], row["away_team"],
                     selection, model_prob, odds, result["suggested_stake"],
                     datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            st.success("Pari enregistre dans le journal (onglet suivant).")

# ---------- onglet 4 : journal de paris ----------

with tab4:
    st.subheader("Journal de tes paris")
    st.caption("Suivi de tes mises reelles pour mesurer ta rentabilite dans le temps.")

    with get_connection() as conn:
        bets_df = pd.read_sql_query("SELECT * FROM bets ORDER BY date DESC", conn)

    if bets_df.empty:
        st.info("Aucun pari enregistre pour l'instant. Utilise l'onglet Calculateur de mise pour en ajouter.")
    else:
        pending = bets_df[bets_df["outcome"] == "pending"]
        if not pending.empty:
            st.markdown("**Paris en attente de resultat :**")
            for _, bet in pending.iterrows():
                cols = st.columns([4, 1, 1])
                cols[0].write(
                    f"{bet['date']} — {bet['home_team']} vs {bet['away_team']} "
                    f"({bet['selection']}, {bet['stake']}€ @ {bet['odds']})"
                )
                if cols[1].button("✅ Gagne", key=f"won_{bet['id']}"):
                    with get_connection() as conn:
                        conn.execute("UPDATE bets SET outcome='won' WHERE id=?", (bet["id"],))
                        conn.commit()
                    st.rerun()
                if cols[2].button("❌ Perdu", key=f"lost_{bet['id']}"):
                    with get_connection() as conn:
                        conn.execute("UPDATE bets SET outcome='lost' WHERE id=?", (bet["id"],))
                        conn.commit()
                    st.rerun()

        settled = bets_df[bets_df["outcome"] != "pending"].copy()
        if not settled.empty:
            settled["gain_net"] = settled.apply(
                lambda r: r["stake"] * (r["odds"] - 1) if r["outcome"] == "won" else -r["stake"], axis=1
            )
            total_stake = settled["stake"].sum()
            total_gain = settled["gain_net"].sum()
            roi = (total_gain / total_stake * 100) if total_stake > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Paris regles", len(settled))
            col2.metric("Gain/perte net", f"{total_gain:+.2f} €")
            col3.metric("ROI", f"{roi:+.1f}%")

            st.dataframe(
                settled[["date", "home_team", "away_team", "selection", "odds", "stake", "outcome", "gain_net"]],
                width="stretch", hide_index=True,
            )
        else:
            st.caption("Aucun pari regle pour l'instant (tous en attente).")
