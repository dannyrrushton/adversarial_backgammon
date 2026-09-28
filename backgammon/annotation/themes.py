"""Detect the strategic themes of a play by comparing the position before and after it."""

from __future__ import annotations

from dataclasses import dataclass

from backgammon.agents.evaluator import features, shots_at
from backgammon.engine import BAR, Board, Play, Player

POINT_NAMES = {
    5: "the 5-point (the 'golden point')",
    4: "the 4-point",
    6: "the 6-point",
    7: "the bar point",
    20: "the opponent's 5-point (the 'golden anchor')",
    21: "the opponent's 4-point",
    18: "the opponent's bar point",
}


def point_name(p: int) -> str:
    return POINT_NAMES.get(p, f"the {p}-point")


@dataclass(frozen=True)
class Theme:
    key: str
    text: str
    weight: float  # how central the theme is to the play; used to order explanations


def play_themes(before: Board, player: Player, play: Play) -> list[Theme]:
    """Themes sorted most important first."""
    after = play.result
    if play.is_pass:
        return [Theme("pass", "No legal move exists, so the turn passes.", 1.0)]
    f0, f1 = features(before, player), features(after, player)
    themes: list[Theme] = []

    if after.winner() is player:
        themes.append(Theme("win", "Bears off the last checker and wins the game.", 10.0))

    hit_points = [m.dst for m in play.moves if m.hit]
    if hit_points:
        where = ", ".join(str(p) for p in hit_points)
        extra = " (a double hit)" if len(hit_points) > 1 else ""
        pips = sum(hit_points)  # a checker hit on our p-point loses p pips
        themes.append(
            Theme(
                "hit",
                f"Hits on {where}{extra}, sending {len(hit_points)} opposing checker"
                f"{'s' * (len(hit_points) > 1)} to the bar and costing the opponent about {pips} pips.",
                3.0 + len(hit_points),
            )
        )

    new_points = sorted(set(f1.made_points) - set(f0.made_points), reverse=True)
    for p in new_points:
        if p >= 19:
            themes.append(Theme("anchor", f"Makes an anchor on {point_name(p)}, a safe landing spot deep in enemy territory.", 2.2))
        elif p <= 7:
            themes.append(Theme("make_point", f"Makes {point_name(p)}, strengthening the blockade in front of the opponent.", 2.5 if p in (4, 5, 7) else 2.0))
        else:
            themes.append(Theme("make_point", f"Makes {point_name(p)}, a useful outfield point for building a prime.", 1.2))

    if len(f1.home_points) > len(f0.home_points):
        themes.append(Theme("home_board", f"Home board now has {len(f1.home_points)} closed points, making it harder to re-enter after a hit.", 1.0))
    if f1.opp_on_bar and len(f1.home_points) == 6:
        themes.append(Theme("closeout", "Closes out the home board: the opponent cannot enter until a point opens.", 4.0))

    if f1.prime >= 3 and f1.prime > f0.prime:
        themes.append(Theme("prime", f"Extends a prime to {f1.prime} consecutive points, trapping checkers behind it.", 1.5 + 0.4 * f1.prime))

    if any(m.src == BAR for m in play.moves):
        themes.append(Theme("enter", "Enters from the bar, which the rules require before anything else.", 1.5))
    escaped = f0.back_checkers - f1.back_checkers
    if escaped > 0:
        themes.append(Theme("escape", f"Escapes {escaped} back checker{'s' * (escaped > 1)} from the opponent's home board.", 1.8))

    slotted = [p for p in (4, 5, 7) if after.count(player, p) == 1 and before.count(player, p) == 0]
    if slotted and f1.contact:
        themes.append(Theme("slot", f"Slots {point_name(slotted[0])}, hoping to cover it next turn to make the point.", 1.1))

    borne = f1.borne_off - f0.borne_off
    if borne and after.winner() is None:
        themes.append(Theme("bear_off", f"Bears off {borne} checker{'s' * (borne > 1)} ({f1.borne_off} of 15 off).", 2.0))

    if f0.contact and not f1.contact:
        lead = f1.pip_lead
        state = f"leading by {lead}" if lead > 0 else f"trailing by {-lead}" if lead < 0 else "level"
        themes.append(Theme("race", f"Breaks contact: the game is now a pure race, {state} pips.", 2.0))

    if f1.contact:
        # Blots in the opponent's home board cost few pips when hit; blots further forward matter.
        exposed = [p for p in f1.blots if p <= 18]
        back = [p for p in f1.blots if p > 18]
        if exposed:
            shots = shots_at(after, player, exposed)
            blots = ", ".join(str(p) for p in exposed)
            plural = "s" * (len(exposed) > 1)
            if shots == 0:
                themes.append(Theme("risk", f"Leaves blot{plural} on {blots}, but no roll can hit {'them' if plural else 'it'}.", 0.3))
            else:
                tone = "a calculated risk" if shots <= 11 else "a real risk"
                themes.append(Theme("risk", f"Leaves blot{plural} on {blots}: {shots}/36 rolls hit ({tone}).", 0.5 + shots / 12))
        if len(back) >= 2 and escaped <= 0 and len(back) > len([p for p in f0.blots if p > 18]):
            themes.append(Theme("split", f"Splits the back checkers ({', '.join(map(str, back))}) to cover more escape and anchor rolls; being hit that far back costs few pips.", 0.9))
        if not f1.blots and f0.blots:
            themes.append(Theme("safe", "Leaves no blots, so there is nothing for the opponent to hit.", 1.3))

    moved_from_mid = sum(1 for m in play.moves if m.src == 13 and 7 <= m.dst <= 11)
    if moved_from_mid and f1.contact:
        themes.append(Theme("builders", "Brings builders down from the midpoint to help make new points.", 0.8))

    if f1.stacked > f0.stacked + 1:
        themes.append(Theme("stack", "Piles extra checkers on one point, which wastes flexibility.", 0.4))

    if not themes:
        pips = before.pip_count(player) - after.pip_count(player)
        themes.append(Theme("quiet", f"A quiet move that advances {pips} pips without changing the structure.", 0.3))
    themes.sort(key=lambda t: t.weight, reverse=True)
    return themes


def blot_exposure(board: Board, player: Player) -> dict[int, int]:
    """Shots (of 36) against each of ``player``'s blots."""
    return {p: shots_at(board, player, [p]) for p in range(1, 25) if board.count(player, p) == 1}

