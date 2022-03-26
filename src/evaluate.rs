use crate::bitboard::*;
use crate::board::*;
use crate::color::*;
use crate::mv::*;
use crate::square::*;
use crate::traversal_path::*;
use core::cmp::Ordering;
use log::debug;
use std::fmt;

const CASTLE_BONUS: i32 = 30;
const OFF_INITIAL_SQUARE_BONUS: i32 = 15;
const OPEN_FILE_BONUS: i32 = 21;

// Relative values of single piece
fn value_of(piece: PieceEnum) -> i32 {
    match piece {
        PAWN => 100,
        KNIGHT => 300,
        BISHOP => 300,
        ROOK => 500,
        QUEEN => 900,
        KING => 100000,
    }
}

// Value of all pieces
fn piece_value(presence: &Presence) -> i32 {
    let mut value: i32 = 0;
    for piece in ALL_PIECES.into_iter() {
        value += presence.for_piece(piece).popcnt() as i32 * value_of(piece)
    }
    value
}

/*
 * Evaluate the position for the given color.
 *
 * Positive values favor white, negative favor black.
 *
 * TODO:
 * - rooks doubled
 * - outposts?
 * - passed pawns?
 * - knight in middle of board
 * - separate scoring for different phases of the game?
 */
pub fn evaluate_position(board: &Board, path: &TraversalPath) -> Result<Score> {
    if board.checkmate(board.color_to_move)? {
        debug!("Found checkmate of {}", BLACK);
        return match board.color_to_move {
            WHITE => Ok(Score::checkmate_white()),
            BLACK => Ok(Score::checkmate_black()),
        };
    }

    let white_value = piece_value(&board.presence_white);
    let black_value = piece_value(&board.presence_black);

    let mut white_bonus: i32 = 0;
    let mut black_bonus: i32 = 0;

    const WHITE_INITIAL_SQUARES: [(Square, PieceEnum); 6] = [
        (A1, ROOK),
        (B1, KNIGHT),
        (C1, BISHOP),
        (F1, BISHOP),
        (G1, KNIGHT),
        (H1, ROOK),
    ];

    const BLACK_INITIAL_SQUARES: [(Square, PieceEnum); 6] = [
        (A8, ROOK),
        (B8, KNIGHT),
        (C8, BISHOP),
        (F8, BISHOP),
        (G8, KNIGHT),
        (H8, ROOK),
    ];

    white_bonus += off_initial_square_bonus(board, WHITE, WHITE_INITIAL_SQUARES);
    black_bonus += off_initial_square_bonus(board, BLACK, BLACK_INITIAL_SQUARES);

    // TODO: only if king is protected
    let castle_bonus = path.fold_left(0, |accum, mv, color| {
        accum
            + match (mv, color) {
                (Move::CastleKingside | Move::CastleQueenside, WHITE) => CASTLE_BONUS,
                (Move::CastleKingside | Move::CastleQueenside, BLACK) => -CASTLE_BONUS,
                _ => 0,
            }
    });

    let open_files = board.open_files();
    white_bonus += open_file_bonus(open_files, board.presence_white.rook);
    black_bonus -= open_file_bonus(open_files, board.presence_black.rook);

    Ok(Score(
        white_value - black_value + white_bonus - black_bonus + castle_bonus,
    ))
}

// TODO: fix this, technically gives a bonus if the piece is captured
fn off_initial_square_bonus(
    board: &Board,
    color: Color,
    initial_squares: [(Square, PieceEnum); 6],
) -> i32 {
    let mut bonus: i32 = 0;
    for (square, piece) in initial_squares {
        if !board.contains_piece(square, piece, color) {
            bonus += OFF_INITIAL_SQUARE_BONUS;
        }
    }
    bonus
}

fn open_file_bonus(open_files: Bitboard, rooks: Bitboard) -> i32 {
    (open_files & rooks).popcnt() as i32 * OPEN_FILE_BONUS
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Score(pub i32);

impl Score {
    pub const MAX: Score = Score(i32::MAX);
    pub const MIN: Score = Score(i32::MIN);
    pub const ZERO: Score = Score(0);

    pub fn checkmate_black() -> Score {
        Score::MAX.minus(1)
    }

    pub fn checkmate_white() -> Score {
        Score::MIN.plus(1)
    }

    pub fn minus(&self, x: i32) -> Score {
        let Score(y) = self;
        Score(y - x)
    }

    pub fn plus(&self, x: i32) -> Score {
        let Score(y) = self;
        Score(y + x)
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        let &Score(i) = self;
        let &Score(j) = other;
        i.cmp(&j)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Score(i) = self;
        write!(f, "{}", i)
    }
}
