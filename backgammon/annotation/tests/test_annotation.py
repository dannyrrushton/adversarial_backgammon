from backgammon.annotation import Annotator, Tutor, grade, legality_notes, play_themes, rule
from backgammon.annotation.annotator import explain_dice_usage
from backgammon.agents.ollama_client import OllamaClient, OllamaError
from backgammon.engine import Board, Player, Roll, legal_plays

W, B = Player.WHITE, Player.BLACK


def find(board, player, roll, notation):
    return next(p for p in legal_plays(board, player, roll) if p.notation() == notation)


def theme_keys(board, player, roll, notation):
    return {t.key for t in play_themes(board, player, find(board, player, roll, notation))}


# --- themes -----------------------------------------------------------------------------------


def test_making_the_five_point():
    keys = theme_keys(Board.initial(), W, Roll(3, 1), "8/5 6/5")
    assert "make_point" in keys and "home_board" in keys


def test_escape_theme():
    assert "escape" in theme_keys(Board.initial(), W, Roll(6, 5), "24/18 18/13")


def test_split_theme():
    assert "split" in theme_keys(Board.initial(), W, Roll(2, 1), "24/22 24/23")


def test_hit_and_risk_themes():
    board = Board.from_relative({8: 3, 6: 5, 13: 5, 24: 2}, {20: 1, 24: 1, 13: 5, 8: 3, 6: 5})
    themes = play_themes(board, W, find(board, W, Roll(3, 1), "8/5* 6/5"))
    assert themes[0].key == "hit"
    assert "5 pips" in themes[0].text


def test_bear_off_and_race_themes():
    board = Board.from_relative({6: 2, 5: 2}, {12: 1}, off=(11, 14))
    assert "bear_off" in theme_keys(board, W, Roll(6, 5), "6/off 5/off")
    contact = Board.from_relative({13: 1, 6: 4}, {14: 1, 6: 4}, off=(10, 10))  # Black on White's 11
    assert "race" in theme_keys(contact, W, Roll(6, 1), "13/7 7/6")


def test_winning_move_theme():
    board = Board.from_relative({1: 2}, {6: 15}, off=(13, 0))
    assert play_themes(board, W, legal_plays(board, W, Roll(2, 1))[0])[0].key == "win"


def test_pass_theme():
    closed = {p: 2 for p in range(1, 7)}
    closed[13] = 3
    board = Board.from_relative({6: 14}, closed, bar=(1, 0), off=(0, 0))
    play = legal_plays(board, W, Roll(6, 6))[0]
    assert play_themes(board, W, play)[0].key == "pass"
    assert "cannot enter" in explain_dice_usage(board, W, Roll(6, 6), play)


# --- legality explanations -------------------------------------------------------------------


def test_legality_mentions_each_rule():
    notes = legality_notes(Board.initial(), W, Roll(3, 1), find(Board.initial(), W, Roll(3, 1), "8/5 6/5"))
    assert "8/5" in notes[0] and "empty" in notes[0]
    assert "already holds 1" in notes[1]
    assert "Both dice" in notes[-1]


def test_legality_bar_entry_and_hit():
    board = Board.from_relative({6: 14}, {4: 1, 6: 14}, bar=(1, 0), off=(0, 0))  # Black blot on White's 21
    play = next(p for p in legal_plays(board, W, Roll(4, 2)) if p.moves[0].dst == 21)
    note = legality_notes(board, W, Roll(4, 2), play)[0]
    assert "must enter" in note and "it is hit and sent to the bar" in note


def test_legality_bear_off_rules():
    board = Board.from_relative({4: 1, 2: 1}, {12: 1}, off=(13, 14))
    notes = legality_notes(board, W, Roll(6, 5), legal_plays(board, W, Roll(6, 5))[0])
    assert any("larger" in n and "highest occupied point" in n for n in notes)
    exact = Board.from_relative({6: 1, 3: 1}, {12: 1}, off=(13, 14))
    notes = legality_notes(exact, W, Roll(6, 3), find(exact, W, Roll(6, 3), "6/off 3/off"))
    assert "matches the 6-point exactly" in notes[0]


def test_legality_larger_die_rule():
    board = Board.from_relative({12: 1}, {23: 2}, off=(14, 13))
    play = legal_plays(board, W, Roll(6, 4))[0]
    assert "larger one, the 6" in legality_notes(board, W, Roll(6, 4), play)[-1]


def test_legality_partial_doubles():
    board = Board.from_relative({10: 1}, {21: 2}, off=(14, 13))
    play = legal_plays(board, W, Roll(3, 3))[0]
    assert "only 1 could be played" in legality_notes(board, W, Roll(3, 3), play)[-1]


# --- annotator ----------------------------------------------------------------------------------


def test_annotation_of_top_play():
    ann = Annotator(depth=0).annotate(Board.initial(), W, Roll(3, 1), find(Board.initial(), W, Roll(3, 1), "8/5 6/5"), reasoning="Classic.")
    assert ann.rank == 1 and ann.equity_loss == 0
    text = ann.to_text()
    assert "White rolls 3-1 and plays 8/5 6/5" in text and "Why it is legal" in text and "Classic." in text


def test_annotation_of_weak_play_names_better_one():
    ann = Annotator(depth=0).annotate(Board.initial(), W, Roll(3, 1), find(Board.initial(), W, Roll(3, 1), "6/3 3/2"))
    assert ann.rank > 1 and ann.equity_loss > 0
    assert "preferred 8/5 6/5" in ann.assessment


def test_grade_bands():
    assert grade(0) == "the engine's top choice"
    assert grade(0.3) == "a mistake"
    assert grade(2) == "a blunder"


class FailingClient(OllamaClient):
    def chat(self, *args, **kwargs):
        raise OllamaError("offline")


class EchoClient(OllamaClient):
    def chat(self, messages, **kwargs):
        assert "coach" in messages[0]["content"]
        return "  Making the 5-point is the best start.  "


def test_llm_commentary_and_fallback():
    play = find(Board.initial(), W, Roll(3, 1), "8/5 6/5")
    ann = Annotator(EchoClient(), depth=0, llm_commentary=True).annotate(Board.initial(), W, Roll(3, 1), play)
    assert ann.commentary == "Making the 5-point is the best start."
    ann = Annotator(FailingClient(), depth=0, llm_commentary=True).annotate(Board.initial(), W, Roll(3, 1), play, reasoning="mine")
    assert ann.commentary == "mine"


# --- tutor --------------------------------------------------------------------------------------


def test_hint_lists_top_plays():
    hint = Tutor(depth=0).hint(Board.initial(), W, Roll(3, 1))
    assert hint.best.notation() == "8/5 6/5"
    assert "1. 8/5 6/5" in hint.to_text()


def test_hint_reminds_about_bar():
    board = Board.from_relative({6: 14}, {6: 15}, bar=(1, 0), off=(0, 0))
    assert "must enter first" in Tutor(depth=0).hint(board, W, Roll(4, 2)).to_text()


def test_critique():
    tutor = Tutor(depth=0)
    good = tutor.critique(Board.initial(), W, Roll(3, 1), find(Board.initial(), W, Roll(3, 1), "8/5 6/5"))
    assert good.text.startswith("Well played")
    bad = tutor.critique(Board.initial(), W, Roll(3, 1), find(Board.initial(), W, Roll(3, 1), "24/21 24/23"))
    assert "stronger play was 8/5 6/5" in bad.text and "5-point" in bad.text


def test_rules_lookup():
    assert "bear them off" in rule("bearing off")
    assert rule("nonsense").startswith("Topics:")
