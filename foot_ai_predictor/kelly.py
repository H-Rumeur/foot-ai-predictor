"""
Calcul de la mise optimale selon le critere de Kelly, a partir de la
probabilite estimee par le modele et de la cote proposee par un bookmaker.

Rappel du principe : le critere de Kelly donne la fraction de bankroll a
miser qui maximise la croissance a long terme, EN SUPPOSANT que ta probabilite
est correcte. Si ton modele est mal calibre, Kelly te fera parier trop gros.
C'est pour ca qu'on utilise presque toujours un "Kelly fractionne"
(25 a 50% du Kelly complet) pour absorber l'incertitude du modele.
"""
from config import DEFAULT_KELLY_FRACTION


def implied_probability(decimal_odds: float) -> float:
    """Probabilite implicite par la cote du bookmaker (avant marge retiree)."""
    if decimal_odds <= 1:
        raise ValueError("La cote decimale doit etre superieure a 1.")
    return 1 / decimal_odds


def kelly_stake_fraction(model_prob: float, decimal_odds: float) -> float:
    """
    Fraction de bankroll suggeree par le critere de Kelly complet.
    Formule : f* = (b*p - q) / b
      b = cote decimale - 1 (gain net pour 1 unite misee)
      p = probabilite estimee de gagner
      q = 1 - p
    Renvoie 0 si le pari n'a pas de valeur (edge negatif ou nul) -- Kelly
    dit explicitement de ne pas parier dans ce cas.
    """
    if not (0 < model_prob < 1):
        raise ValueError("model_prob doit etre strictement entre 0 et 1.")
    b = decimal_odds - 1
    q = 1 - model_prob
    f_star = (b * model_prob - q) / b
    return max(f_star, 0.0)


def evaluate_bet(model_prob: float, decimal_odds: float, bankroll: float,
                  kelly_fraction: float = DEFAULT_KELLY_FRACTION) -> dict:
    """
    Evalue un pari potentiel : edge par rapport au marche, et mise suggeree.

    kelly_fraction : quelle part du Kelly complet appliquer (0.3 = 30%, recommande).
    """
    implied_p = implied_probability(decimal_odds)
    edge = model_prob - implied_p  # edge positif = le modele est plus optimiste que le marche

    full_kelly = kelly_stake_fraction(model_prob, decimal_odds)
    applied_fraction = full_kelly * kelly_fraction
    suggested_stake = round(bankroll * applied_fraction, 2)

    return {
        "model_prob": round(model_prob, 4),
        "implied_prob_bookmaker": round(implied_p, 4),
        "edge": round(edge, 4),
        "has_value": edge > 0,
        "full_kelly_fraction": round(full_kelly, 4),
        "applied_kelly_fraction": round(applied_fraction, 4),
        "suggested_stake": suggested_stake,
        "expected_value_per_unit": round(model_prob * (decimal_odds - 1) - (1 - model_prob), 4),
    }
