# Foot AI Predictor

Outil personnel d'aide a la decision pour les paris sportifs sur le foot.
Il calcule des probabilites de victoire/nul/defaite pour 6 grands championnats
europeens a partir d'un modele statistique (regression de Poisson), et t'aide
a calculer une mise raisonnable si tu detectes de la valeur par rapport aux
cotes d'un bookmaker.

**Important :** cet outil ne place aucun pari a ta place. Il t'aide juste a
prendre une decision plus informee. Tu paries ensuite toi-meme, chez le
bookmaker de ton choix, avec ton propre jugement. Ce n'est pas un conseil
financier, et rien ne garantit que ce modele est rentable a long terme --
lis la section "Limites a garder en tete" avant de l'utiliser serieusement.

## Comment ca marche

- **Historique des matchs** (pour entrainer et evaluer le modele) : telecharge
  gratuitement depuis [football-data.co.uk](https://www.football-data.co.uk),
  plusieurs saisons, sans cle API necessaire.
- **Matchs a venir** : recuperes via l'API gratuite
  [football-data.org](https://www.football-data.org) (necessite une cle
  gratuite, voir installation ci-dessous).
- **Modele** : une regression de Poisson par ligue, qui estime une "force
  d'attaque" et une "force de defense" pour chaque equipe, puis en deduit
  le nombre de buts attendus et les probabilites 1X2 pour chaque match.
- **Interface** : un dashboard Streamlit local (dans ton navigateur, mais
  rien ne sort de ta machine).

Les 6 championnats suivis par defaut : Premier League, La Liga, Bundesliga,
Serie A, Ligue 1, Eredivisie. Modifiable dans `config.py`.

## Installation

### 1. Prerequis

- Python 3.10 ou plus recent installe sur ta machine
- Une connexion internet (pour telecharger les donnees)

### 2. Installer les dependances

Dans un terminal, a la racine du projet :

```bash
python -m venv venv
source venv/bin/activate        # sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Obtenir ta cle API gratuite

1. Va sur https://www.football-data.org/client/register
2. Cree un compte gratuit (email + mot de passe, aucune carte bancaire demandee)
3. Recupere ta cle API depuis ton espace client
4. Copie le fichier `.env.example` en `.env` :
   ```bash
   cp .env.example .env
   ```
5. Ouvre `.env` et colle ta cle a la place de `colle_ta_cle_ici`

### 4. Premier lancement du pipeline

Toujours dans le terminal, a la racine du projet :

```bash
python pipeline.py run-all
```

Ca va, dans l'ordre :
1. Creer la base de donnees locale (`data/football.db`)
2. Telecharger l'historique de plusieurs saisons pour les 6 ligues (peut
   prendre 1-2 minutes)
3. Recuperer les matchs a venir via l'API (respecte automatiquement la
   limite de 10 requetes/minute du plan gratuit, donc ca prend ~1 minute)
4. Entrainer un modele par ligue et generer les predictions
5. Lancer un backtest pour verifier la fiabilite du modele sur l'historique

Si tu preferes lancer les etapes une par une (utile pour deboguer) :

```bash
python pipeline.py init-db
python pipeline.py fetch-historical
python pipeline.py fetch-live
python pipeline.py train
python pipeline.py backtest
```

### 5. Lancer le dashboard

```bash
streamlit run app.py
```

Ca ouvre automatiquement un onglet dans ton navigateur sur `localhost:8501`.

## Utilisation au quotidien

Une fois l'installation faite, pour garder les predictions a jour avant
chaque journee de championnat :

```bash
python pipeline.py fetch-live
python pipeline.py train
```

(Le `train` regenere aussi les predictions pour les nouveaux matchs a venir.)

Pense aussi a relancer `fetch-historical` de temps en temps (une fois par
mois par exemple) pour que le modele integre les resultats les plus recents
dans son historique d'entrainement.

## Structure du projet

```
config.py              # ligues suivies, saisons, parametres du modele
db.py                   # acces a la base de donnees SQLite
fetch_historical.py     # telechargement de l'historique (CSV)
fetch_live.py           # recuperation des matchs a venir (API)
team_mapping.py         # fait correspondre les noms d'equipes entre les 2 sources
team_mapping.json       # table de correspondance (editable a la main)
model.py                # le modele de prediction (regression de Poisson)
backtest.py             # evaluation de la fiabilite du modele
kelly.py                # calcul de la mise optimale (critere de Kelly)
pipeline.py             # CLI pour lancer toutes les etapes
app.py                  # dashboard Streamlit
data/football.db        # base de donnees (creee au premier lancement)
data/model.pkl           # modeles entraines sauvegardes
```

## Depannage

**"Equipe inconnue du modele"** : les deux sources de donnees nomment parfois
les equipes differemment (ex: "Manchester City FC" vs "Man City"). Ouvre
`team_mapping.json` et ajoute la correspondance manquante -- le format est
`"Nom cote football-data.org": "Nom cote football-data.co.uk"`.

**"Aucune donnee importee" pendant fetch-historical** : le site
football-data.co.uk a peut-etre change sa structure d'URL depuis l'ecriture
de ce projet. Verifie manuellement dans ton navigateur si
`https://www.football-data.co.uk/mmz4281/2425/E0.csv` telecharge bien un
fichier CSV. Si l'URL a change, ajuste `FOOTBALL_DATA_CO_UK_BASE_URL` dans
`config.py`.

**Erreur 429 / limite de requetes** pendant `fetch-live` : le plan gratuit de
football-data.org est limite a 10 requetes/minute. Le script gere deja des
pauses automatiques, mais si l'erreur persiste, attends une minute et relance.

**"Pas assez de matchs pour entrainer un modele fiable"** : augmente le
nombre de saisons dans `HISTORICAL_SEASONS` (`config.py`), ou verifie que
`fetch-historical` a bien importe des donnees pour cette ligue.

## Pour aller plus loin

- **Ajouter un championnat** : ajoute une entree dans `LEAGUES` (`config.py`)
  avec le code football-data.org et le code football-data.co.uk correspondants.
- **Ajuster l'agressivite des mises** : modifie `DEFAULT_KELLY_FRACTION` dans
  `config.py` (0.3 = 30% du Kelly complet, plus prudent ; 1.0 = Kelly complet,
  plus agressif et plus volatile).
- **Changer d'algorithme** : `model.py` est volontairement isole du reste --
  tu peux y remplacer la regression de Poisson par un autre modele (Elo,
  gradient boosting...) sans toucher au reste du pipeline.

## Limites a garder en tete

- Les cotes des bookmakers integrent deja beaucoup d'information et sont
  ajustees par des professionnels. Detecter un edge reel et consistant est
  difficile, meme avec un bon modele -- regarde toujours l'onglet
  "Performance du modele" avant de faire confiance a une prediction.
- Le modele ne connait que les buts marques/encaisses. Il ignore les
  blessures, suspensions, enjeux du match, meteo, etc. -- des facteurs qui
  peuvent significativement changer l'issue d'un match.
- Un backtest positif sur le passe ne garantit pas des resultats futurs.
- Jouer comporte des risques. Fixe-toi des limites de mise que tu ne
  depasseras pas, et ne mise jamais plus que ce que tu es pret a perdre.
