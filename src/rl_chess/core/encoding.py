"""AlphaZero chess move and board state encoding / decoding.

Action space: 8x8x73 = 4672 discrete actions.
- Planes 0..55: Queen-like moves (8 directions x 7 squares distance)
- Planes 56..63: Knight jumps (8 knight moves)
- Planes 64..72: Underpromotions to Knight, Bishop, Rook (3 directions x 3 pieces)
Queen promotions are encoded as regular distance=1 queen moves to the back rank.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import chess
import numpy as np

# Dimensions
BOARD_SIZE = 8
NUM_MOVE_TYPES = 73
ACTION_SPACE_SIZE = BOARD_SIZE * BOARD_SIZE * NUM_MOVE_TYPES  # 4672

# Directions for queen-like moves: (d_rank, d_file)
# Index: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
QUEEN_DIRECTIONS: List[Tuple[int, int]] = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]

# Knight jump deltas: (d_rank, d_file)
KNIGHT_DELTAS: List[Tuple[int, int]] = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
]

# Underpromotions: piece -> base plane index (offset by direction: -1, 0, +1 file)
UNDERPROMOTIONS = {
    chess.KNIGHT: 64,
    chess.BISHOP: 67,
    chess.ROOK: 70,
}

# Number of channels for board representation (with T=4 history)
# 4 history steps * 12 piece channels = 48
# 4 history steps * 2 repetition channels = 8
# 7 constant / auxiliary channels:
#   color (1), move count (1), player castling K (1), player castling Q (1),
#   opponent castling K (1), opponent castling Q (1), halfmove clock (1)
NUM_INPUT_CHANNELS = 48 + 8 + 7  # 63


def mirror_square(sq: chess.Square) -> chess.Square:
    """Flip rank (0..7 -> 7..0) for black's perspective, keeping file unchanged."""
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    return chess.square(file, 7 - rank)


def orient_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """Orient move so current player always plays as White."""
    if turn == chess.WHITE:
        return move
    return chess.Move(
        from_square=mirror_square(move.from_square),
        to_square=mirror_square(move.to_square),
        promotion=move.promotion,
    )


def unorient_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """Convert an oriented move back to absolute coordinates."""
    if turn == chess.WHITE:
        return move
    return chess.Move(
        from_square=mirror_square(move.from_square),
        to_square=mirror_square(move.to_square),
        promotion=move.promotion,
    )


def move_to_plane_index(move: chess.Move) -> int:
    """Compute the 0..72 plane index for an oriented move."""
    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    d_rank = to_rank - from_rank
    d_file = to_file - from_file

    # 1. Underpromotions (Knight, Bishop, Rook)
    if move.promotion in UNDERPROMOTIONS:
        base = UNDERPROMOTIONS[move.promotion]
        # d_file is in {-1, 0, 1}
        col_offset = d_file + 1  # -1 -> 0, 0 -> 1, +1 -> 2
        return base + col_offset

    # 2. Knight moves
    for idx, (kr, kf) in enumerate(KNIGHT_DELTAS):
        if d_rank == kr and d_file == kf:
            return 56 + idx

    # 3. Queen-like moves (distance 1..7 in 8 directions)
    for dir_idx, (dr, df) in enumerate(QUEEN_DIRECTIONS):
        # Check if (d_rank, d_file) is a positive multiple of (dr, df)
        for dist in range(1, 8):
            if d_rank == dr * dist and d_file == df * dist:
                return dir_idx * 7 + (dist - 1)

    raise ValueError(f"Unrecognized chess move geometry: {move} (d_rank={d_rank}, d_file={d_file})")


def plane_index_to_move(
    from_sq: chess.Square,
    plane_idx: int,
    is_pawn: bool = False,
) -> chess.Move:
    """Decode from_square and plane_idx back to an oriented chess.Move."""
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)

    # 1. Underpromotions (64..72)
    if plane_idx >= 64:
        for piece, base in UNDERPROMOTIONS.items():
            if base <= plane_idx < base + 3:
                d_file = (plane_idx - base) - 1
                to_rank = 7  # Must be 8th rank
                to_file = from_file + d_file
                to_sq = chess.square(to_file, to_rank)
                return chess.Move(from_sq, to_sq, promotion=piece)
        raise ValueError(f"Invalid underpromotion plane index: {plane_idx}")

    # 2. Knight moves (56..63)
    if plane_idx >= 56:
        k_idx = plane_idx - 56
        dr, df = KNIGHT_DELTAS[k_idx]
        to_rank = from_rank + dr
        to_file = from_file + df
        to_sq = chess.square(to_file, to_rank)
        return chess.Move(from_sq, to_sq)

    # 3. Queen-like moves (0..55)
    dir_idx = plane_idx // 7
    dist = (plane_idx % 7) + 1
    dr, df = QUEEN_DIRECTIONS[dir_idx]
    to_rank = from_rank + dr * dist
    to_file = from_file + df * dist
    to_sq = chess.square(to_file, to_rank)

    # In chess, a move promotes to Queen if and only if the moving piece is a pawn reaching rank 7 (8th rank)
    promotion = chess.QUEEN if (is_pawn and to_rank == 7) else None
    return chess.Move(from_sq, to_sq, promotion=promotion)


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    """Encode an absolute chess.Move into an integer action index in [0, 4671]."""
    oriented = orient_move(move, turn)
    from_sq = oriented.from_square
    plane_idx = move_to_plane_index(oriented)
    return from_sq * NUM_MOVE_TYPES + plane_idx


def decode_action(
    action: int,
    turn: Optional[chess.Color] = None,
    board: Optional[chess.Board] = None,
    is_pawn: bool = False,
) -> chess.Move:
    """Decode an integer action index [0, 4671] into an absolute chess.Move.
    
    If `board` is supplied, piece type and turn are inferred automatically.
    """
    if board is not None:
        turn = board.turn
        from_sq_oriented = action // NUM_MOVE_TYPES
        abs_from_sq = mirror_square(from_sq_oriented) if turn == chess.BLACK else from_sq_oriented
        is_pawn = (board.piece_type_at(abs_from_sq) == chess.PAWN)
    elif turn is None:
        raise ValueError("Must provide either board or turn to decode_action")

    from_sq = action // NUM_MOVE_TYPES
    plane_idx = action % NUM_MOVE_TYPES
    oriented = plane_index_to_move(from_sq, plane_idx, is_pawn=is_pawn)
    return unorient_move(oriented, turn)


def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """Returns a boolean array of shape (4672,) where True indicates a legal action."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    turn = board.turn
    for move in board.legal_moves:
        idx = encode_move(move, turn)
        mask[idx] = True
    return mask


def encode_board(
    board: chess.Board,
    history: Optional[List[chess.Board]] = None,
    num_history_steps: int = 4,
) -> np.ndarray:
    """Encode chess board into an (NUM_INPUT_CHANNELS, 8, 8) float32 tensor.
    
    The board is oriented to the current player's perspective:
    - If White to move: rank 1 is rank 0, pieces as-is.
    - If Black to move: rank 8 is rank 0, pieces colors inverted.
    """
    planes = np.zeros((NUM_INPUT_CHANNELS, 8, 8), dtype=np.float32)
    current_turn = board.turn

    # Prepare board history list ending with current board
    boards: List[chess.Board] = []
    if history:
        boards.extend(history[-(num_history_steps - 1):])
    boards.append(board)

    # Pad with copies of oldest board if history < num_history_steps
    while len(boards) < num_history_steps:
        boards.insert(0, boards[0])

    channel = 0
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    for b in boards:
        # 12 piece planes: 6 current player pieces, then 6 opponent pieces
        for color in [current_turn, not current_turn]:
            for p_type in piece_types:
                for sq in b.pieces(p_type, color):
                    r = chess.square_rank(sq) if current_turn == chess.WHITE else 7 - chess.square_rank(sq)
                    f = chess.square_file(sq)
                    planes[channel, r, f] = 1.0
                channel += 1

        # 2 repetition planes: 1st repetition, 2nd repetition
        # Count occurrences in board position history
        rep_count = b.is_repetition(2) + b.is_repetition(3)
        if rep_count >= 1:
            planes[channel, :, :] = 1.0
        channel += 1
        if rep_count >= 2:
            planes[channel, :, :] = 1.0
        channel += 1

    # 7 constant / auxiliary planes
    # 1. Turn color (all 1s for White, all 0s for Black)
    if current_turn == chess.WHITE:
        planes[channel, :, :] = 1.0
    channel += 1

    # 2. Total move count (normalized)
    planes[channel, :, :] = min(1.0, board.fullmove_number / 100.0)
    channel += 1

    # 3 & 4. Current player castling rights (kingside, queenside)
    if current_turn == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[channel, :, :] = 1.0
        channel += 1
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[channel, :, :] = 1.0
        channel += 1
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[channel, :, :] = 1.0
        channel += 1
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[channel, :, :] = 1.0
        channel += 1

    # 5 & 6. Opponent castling rights (kingside, queenside)
    opp_color = not current_turn
    if opp_color == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[channel, :, :] = 1.0
        channel += 1
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[channel, :, :] = 1.0
        channel += 1
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[channel, :, :] = 1.0
        channel += 1
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[channel, :, :] = 1.0
        channel += 1

    # 7. Halfmove clock (rule of 50-move draw)
    planes[channel, :, :] = min(1.0, board.halfmove_clock / 100.0)
    channel += 1

    assert channel == NUM_INPUT_CHANNELS, f"Expected {NUM_INPUT_CHANNELS} channels, got {channel}"
    return planes
