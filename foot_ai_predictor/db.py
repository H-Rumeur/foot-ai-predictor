"""
Gestion de la base de donnees SQLite.
Toutes les autres parties du projet passent par ce module pour lire/ecrire
des donnees -- ca evite de dupliquer les requetes SQL partout.
"""
import sqlite3
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    season TEXT NOT NULL,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_goals INTEGER NOT NULL,
    away_goals INTEGER NOT NULL,
    result TEXT NOT NULL,          -- 'H', 'D', ou 'A'
    b365h REAL,                    -- cote Bet365 victoire domicile (si dispo)
    b365d REAL,
    b365a REAL,
    UNIQUE(league, date, home_team, away_team)
);

CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fdorg_id INTEGER UNIQUE,       -- id cote football-data.org, evite les doublons
    league TEXT NOT NULL,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    status TEXT NOT NULL           -- 'SCHEDULED', 'FINISHED', etc.
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    league TEXT NOT NULL,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    lambda_home REAL NOT NULL,     -- buts attendus domicile
    lambda_away REAL NOT NULL,     -- buts attendus exterieur
    prob_home REAL NOT NULL,
    prob_draw REAL NOT NULL,
    prob_away REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(fixture_id) REFERENCES fixtures(id)
);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    league TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    selection TEXT NOT NULL,       -- 'home', 'draw', ou 'away'
    model_prob REAL NOT NULL,
    odds REAL NOT NULL,
    stake REAL NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'won', 'lost'
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_connection():
    """Fournit une connexion SQLite et la ferme proprement a la fin."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Cree les tables si elles n'existent pas deja. A appeler une seule fois."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    print(f"Base de donnees initialisee : {DB_PATH}")


def insert_match(conn, league, season, date, home_team, away_team,
                  home_goals, away_goals, result, b365h=None, b365d=None, b365a=None):
    """Insere un match historique. Ignore silencieusement les doublons."""
    conn.execute(
        """
        INSERT OR IGNORE INTO matches
            (league, season, date, home_team, away_team, home_goals, away_goals,
             result, b365h, b365d, b365a)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (league, season, date, home_team, away_team, home_goals, away_goals,
         result, b365h, b365d, b365a),
    )


def insert_fixture(conn, fdorg_id, league, date, home_team, away_team, status):
    """Insere ou met a jour un match a venir."""
    conn.execute(
        """
        INSERT INTO fixtures (fdorg_id, league, date, home_team, away_team, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(fdorg_id) DO UPDATE SET
            date=excluded.date, status=excluded.status
        """,
        (fdorg_id, league, date, home_team, away_team, status),
    )


def get_matches_df(league=None):
    """Retourne l'historique des matchs sous forme de DataFrame pandas."""
    import pandas as pd
    with get_connection() as conn:
        if league:
            return pd.read_sql_query(
                "SELECT * FROM matches WHERE league = ? ORDER BY date", conn, params=(league,)
            )
        return pd.read_sql_query("SELECT * FROM matches ORDER BY date", conn)


def get_distinct_teams(league=None):
    """
    Renvoie la liste triee des noms d'equipes tels qu'ils existent vraiment
    dans l'historique (donc tels qu'utilises par football-data.co.uk).
    Utile pour corriger team_mapping.json : compare ces noms a ceux affiches
    par les messages "Equipe inconnue du modele" pendant l'entrainement.
    """
    with get_connection() as conn:
        if league:
            rows = conn.execute(
                "SELECT DISTINCT home_team AS team FROM matches WHERE league = ? "
                "UNION SELECT DISTINCT away_team FROM matches WHERE league = ? "
                "ORDER BY team",
                (league, league),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT home_team AS team FROM matches "
                "UNION SELECT DISTINCT away_team FROM matches ORDER BY team"
            ).fetchall()
        return [r["team"] for r in rows]


def get_upcoming_fixtures_df(league=None):
    import pandas as pd
    with get_connection() as conn:
        query = "SELECT * FROM fixtures WHERE status = 'SCHEDULED'"
        params = ()
        if league:
            query += " AND league = ?"
            params = (league,)
        query += " ORDER BY date"
        return pd.read_sql_query(query, conn, params=params)


def save_prediction(conn, fixture_id, league, date, home_team, away_team,
                     lambda_home, lambda_away, prob_home, prob_draw, prob_away):
    from datetime import datetime, timezone
    # supprime l'ancienne prediction pour ce match si elle existe deja, pour
    # ne pas accumuler des doublons a chaque fois qu'on rafraichit le modele
    conn.execute("DELETE FROM predictions WHERE fixture_id = ?", (fixture_id,))
    conn.execute(
        """
        INSERT INTO predictions
            (fixture_id, league, date, home_team, away_team,
             lambda_home, lambda_away, prob_home, prob_draw, prob_away, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (fixture_id, league, date, home_team, away_team,
         lambda_home, lambda_away, prob_home, prob_draw, prob_away,
         datetime.now(timezone.utc).isoformat()),
    )
