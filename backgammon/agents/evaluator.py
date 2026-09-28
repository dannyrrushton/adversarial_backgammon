"""Hand-written position evaluation.

The LLM is a poor backgammon calculator, so agents use this evaluator (plus a one-ply lookahead in
:mod:`backgammon.agents.search`) to shortlist plays, then let the model choose and explain. The
same features drive the strategy text in :mod:`backgammon.annotation`.

All values are from the point of view of ``player`` *after* their move, with the opponent to roll.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backgammon.engine import ALL_ROLLS, BAR, Board, Player
from backgammon.engine.board import opponent_point
from backgammon.engine.game import classify_win

# How much a made point is worth by relative point number: the 5, 4, 6 and 7 ("bar") points are
# the most valuable for blocking and containing.
POINT_VALUE = {1: 0.3, 2: 0.5, 3: 0.7, 4: 1.1, 5: 1.4, 6: 1.2, 7: 0.9, 8: 0.6, 9: 0.4}


@dataclass(frozen=True)
class Features:
    pips: int
    opp_pips: int
    blots: tuple[int, ...]  # relative points with exactly one checker
    shots: int  # opponent rolls (of 36) that hit at least one of our blots
    home_points: tuple[int, ...]  # made points in our home board
    made_points: tuple[int, ...]  # every point with 2+ checkers
    prime: int  # longest run of consecutive made points
    anchors: tuple[int, ...]  # made points in the opponent's home board (our 19..24)
    back_checkers: int  # checkers on our 19..24 or the bar
    on_bar: int
    opp_on_bar: int
    borne_off: int
    opp_borne_off: int
    stacked: int  # checkers beyond the third on any point
    contact: bool

    @property
    def pip_lead(self) -> int:
        return self.opp_pips - self.pips


def made_points(board: Board, player: Player) -> list[int]:
    return [p for p in range(1, 25) if board.count(player, p) >= 2]


def longest_prime(points: list[int]) -> int:
    best = run = 0
    prev = None
    for p in sorted(points):
        run = run + 1 if prev is not None and p == prev + 1 else 1
        best = max(best, run)
        prev = p
    return best


def shots_at(board: Board, player: Player, targets: list[int] | None = None) -> int:
    """How many of the opponent's 36 rolls hit at least one of ``player``'s blots.

    Direct and combination shots are counted; a combination needs its intermediate landing
    points open. Checkers on the opponent's bar must enter first, so only entering shots count
    for them.
    """
    blots = targets if targets is not None else [p for p in range(1, 25) if board.count(player, p) == 1]
    if not blots:
        return 0
    opp = player.opponent
    # Blots in the opponent's numbering.
    blot_spots = {opponent_point(p) for p in blots}
    sources = [BAR] if board.bar[opp] else board.occupied_points(opp)
    blocked = {opponent_point(p) for p in range(1, 25) if board.count(player, p) >= 2}

    def reaches(src: int, steps: list[int]) -> bool:
        pos = src
        for step in steps:
            pos -= step
            if pos < 1:
                return False
            if pos in blot_spots:
                return True
            if pos in blocked:
                return False
        return False

    total = 0
    for roll in ALL_ROLLS:
        if roll.is_double:
            paths = [[roll.d1] * k for k in range(1, 5)]
        else:
            paths = [[roll.d1], [roll.d2], [roll.d1, roll.d2], [roll.d2, roll.d1]]
        if any(reaches(src, path) for src in sources for path in paths):
            total += 1 if roll.is_double else 2
    return total


def features(board: Board, player: Player) -> Features:
    opp = player.opponent
    made = made_points(board, player)
    contact = board.has_contact()
    blots = tuple(p for p in range(1, 25) if board.count(player, p) == 1)
    return Features(
        pips=board.pip_count(player),
        opp_pips=board.pip_count(opp),
        blots=blots,
        shots=shots_at(board, player, list(blots)) if contact else 0,
        home_points=tuple(p for p in made if p <= 6),
        made_points=tuple(made),
        prime=longest_prime(made),
        anchors=tuple(p for p in made if p >= 19),
        back_checkers=board.bar[player] + sum(board.count(player, p) for p in range(19, 25)),
        on_bar=board.bar[player],
        opp_on_bar=board.bar[opp],
        borne_off=board.off[player],
        opp_borne_off=board.off[opp],
        stacked=sum(max(0, board.count(player, p) - 3) for p in range(1, 25)),
        contact=contact,
    )


def race_win_probability(pips: int, opp_pips: int, on_roll: bool) -> float:
    """Normal approximation to a pure race: being on roll is worth about four pips."""
    lead = opp_pips - pips + (4 if on_roll else -4)
    sigma = 0.2 * (pips + opp_pips) / 2 + 2
    return 0.5 * (1 + math.erf(lead / (sigma * math.sqrt(2))))


def _structure(board: Board, player: Player, f: Features, g: Features) -> float:
    """Positional score for ``player`` (features ``f``) against an opponent with features ``g``."""
    score = sum(POINT_VALUE.get(p, 0.2) for p in f.made_points)
    # A prime only matters with opponent checkers stuck behind it.
    if g.back_checkers:
        score += 0.25 * f.prime**2
    score += 0.6 * len(f.anchors)
    score -= 0.12 * f.stacked
    # Checkers on the bar hurt more the stronger the board they must enter against.
    score += g.on_bar * (1.2 + 0.35 * len(f.home_points))
    return score


def _blot_risk(board: Board, player: Player, f: Features, g: Features) -> float:
    """Expected damage from our blots being hit on the opponent's next roll."""
    if not f.blots or not f.contact:
        return 0.0
    risk = 0.0
    for p in f.blots:
        frac = shots_at(board, player, [p]) / 36
        loss = (25 - p) / 25  # pips thrown away by a checker hit on point p
        risk += frac * (1.0 + 3.0 * loss) * (1 + 0.25 * len(g.home_points))
    return risk


def equity(board: Board, player: Player) -> float:
    """Cubeless equity estimate for ``player`` in [-3, 3], opponent to roll. Used to rank plays."""
    winner = board.winner()
    if winner is not None:
        value = classify_win(board, winner).value
        return value if winner is player else -value
    f = features(board, player)
    if not f.contact:
        p = race_win_probability(f.pips, f.opp_pips, on_roll=False)
        return 2 * p - 1
    return 2 * _play_probability(board, player, f) - 1 + gammon_bonus(f)


def score_components(board: Board, player: Player, f: Features | None = None) -> tuple[float, float, float, float]:
    """(pip lead net of the roll, structure difference, blot risk, borne-off difference)."""
    f = f or features(board, player)
    g = features(board, player.opponent)
    return (
        f.pip_lead - 8,  # the opponent is on roll: roughly an 8 pip swing
        _structure(board, player, f, g) - _structure(board, player.opponent, g, f),
        _blot_risk(board, player, f, g),
        f.borne_off - f.opp_borne_off,
    )


# Hand-tuned for choosing moves: blot safety is weighted heavily because leaving a shot is the
# commonest tactical error, even though one turn's exposure barely moves the final result.
PLAY_WEIGHTS = (0.07, 0.35, -0.9, 0.15, 0.0)

# Fitted by logistic regression on self-play outcomes (``python -m backgammon.agents.calibrate``)
# so that the probabilities shown to players and used for cube decisions are calibrated.
CALIBRATED_WEIGHTS = (0.022, 0.154, -0.021, 0.271, 0.004)


def _logistic(components, weights) -> float:
    z = sum(c * w for c, w in zip(components, weights)) + weights[-1]
    return 1 / (1 + math.exp(-z))


def _play_probability(board: Board, player: Player, f: Features) -> float:
    return _logistic(score_components(board, player, f), PLAY_WEIGHTS)


def win_probability(board: Board, player: Player, f: Features | None = None) -> float:
    """Calibrated win chance for ``player`` with the opponent on roll."""
    if board.winner() is not None:
        return 1.0 if board.winner() is player else 0.0
    f = f or features(board, player)
    if not f.contact:
        return race_win_probability(f.pips, f.opp_pips, on_roll=False)
    return _logistic(score_components(board, player, f), CALIBRATED_WEIGHTS)


def gammon_bonus(f: Features) -> float:
    """Small extra equity for gammon chances when the opponent is trapped or on the bar."""
    bonus = 0.0
    if f.opp_borne_off == 0 and f.borne_off > 0:
        bonus += 0.1 + 0.02 * f.borne_off
    if f.opp_on_bar and len(f.home_points) >= 5:
        bonus += 0.15 * f.opp_on_bar
    if f.borne_off == 0 and f.opp_borne_off > 0:
        bonus -= 0.1 + 0.02 * f.opp_borne_off
    return bonus

