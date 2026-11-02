"""
Point d'entree unique pour toutes les etapes du pipeline.
Lance chaque commande dans cet ordre la premiere fois :

    python pipeline.py init-db
    python pipeline.py fetch-historical
    python pipeline.py fetch-live
    python pipeline.py train
    python pipeline.py backtest

Ensuite, au quotidien, il suffit de relancer fetch-live + train pour garder
les predictions a jour avec les derniers resultats et matchs programmes.
"""
import argparse
import sys
from datetime import datetime

from config import LEAGUES
from db import (
    init_db, get_connection, get_matches_df, get_upcoming_fixtures_df,
    save_prediction, get_distinct_teams,
)
from fetch_historical import import_all_history
from fetch_live import import_all_upcoming
from model import train_all_leagues, PoissonMatchModel, MODEL_PATH
from backtest import run_backtest, print_backtest_report
from team_mapping import to_historical_name
from refresh import retrain_and_predict


def cmd_init_db(args):
    init_db()


def cmd_fetch_historical(args):
    import_all_history()


def cmd_fetch_live(args):
    import_all_upcoming(days_ahead=getattr(args, "days", 14))


def cmd_train(args):
    print("Entrainement des modeles (un par ligue)...")
    result = retrain_and_predict()
    for lk in result["trained_leagues"]:
        print(f"  {lk} : modele entraine")
    for lk in result["skipped_leagues"]:
        print(f"  [!] {lk} : aucune donnee ou pas assez de matchs, lance d'abord fetch-historical")
    print(f"\n{result['n_predictions']} predictions generees.")
    if result["unknown_matches"]:
        print(f"{len(result['unknown_matches'])} match(s) ignores (equipe inconnue du modele) :")
        for m in result["unknown_matches"]:
            print(f"  [!] {m}")


def cmd_list_teams(args):
    """
    Affiche les noms d'equipes tels que stockes dans l'historique (donc tels
    qu'utilises par football-data.co.uk). Compare ces noms a ceux affiches par
    les messages "Equipe inconnue du modele" pour corriger team_mapping.json.
    """
    if args.league and args.league not in LEAGUES:
        print(f"Ligue inconnue : '{args.league}'. Choix possibles : {list(LEAGUES.keys())}")
        return
    teams = get_distinct_teams(args.league)
    label = LEAGUES[args.league]["display_name"] if args.league else "toutes ligues confondues"
    print(f"{len(teams)} equipes trouvees dans l'historique ({label}) :\n")
    for t in teams:
        print(f"  {t}")


def cmd_backtest(args):
    for league_key in LEAGUES:
        matches = get_matches_df(league_key)
        if matches.empty:
            print(f"[!] {league_key} : aucune donnee, lance d'abord fetch-historical")
            continue
        result = run_backtest(matches, league_key)
        print_backtest_report(result)


def cmd_run_all(args):
    """Enchaine toutes les etapes d'un coup, pratique pour la premiere installation."""
    cmd_init_db(args)
    cmd_fetch_historical(args)
    cmd_fetch_live(args)
    cmd_train(args)
    cmd_backtest(args)


def main():
    parser = argparse.ArgumentParser(description="Pipeline du predicteur foot")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Cree les tables de la base de donnees")
    sub.add_parser("fetch-historical", help="Telecharge l'historique (CSV football-data.co.uk)")

    p_live = sub.add_parser("fetch-live", help="Recupere les matchs a venir (API football-data.org)")
    p_live.add_argument("--days", type=int, default=14)

    sub.add_parser("train", help="Entraine les modeles et genere les predictions")

    p_list = sub.add_parser("list-teams", help="Liste les noms d'equipes reels de l'historique (debogage mapping)")
    p_list.add_argument("--league", type=str, default=None,
                         help="Ex: la_liga, bundesliga... (toutes les ligues si omis)")
    sub.add_parser("backtest", help="Evalue la fiabilite du modele sur l'historique")
    sub.add_parser("run-all", help="Enchaine toutes les etapes (premiere installation)")

    args = parser.parse_args()

    commands = {
        "init-db": cmd_init_db,
        "fetch-historical": cmd_fetch_historical,
        "fetch-live": cmd_fetch_live,
        "train": cmd_train,
        "list-teams": cmd_list_teams,
        "backtest": cmd_backtest,
        "run-all": cmd_run_all,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
