import chess

from rl_chess.game import format_material_balance, material_balance_white


def test_material_balance_start_position():
    board = chess.Board()
    assert material_balance_white(board) == 0
    assert format_material_balance(board) == "Material: even"


def test_material_balance_missing_queen():
    board = chess.Board()
    board.remove_piece_at(chess.D8)
    assert material_balance_white(board) == 9
    assert format_material_balance(board) == "White +9"
    board.remove_piece_at(chess.D1)
    assert material_balance_white(board) == 0
    board.remove_piece_at(chess.E2)
    assert material_balance_white(board) == -1
    assert format_material_balance(board) == "Black +1"
