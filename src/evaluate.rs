use crate::bitboard::*;
use crate::board::*;
use crate::color::*;
#[cfg(test)]
use crate::fen;
use crate::mv::*;
use crate::square::*;
use crate::traversal_path::*;
use core::cmp::Ordering;
use log::debug;
use std::fmt;

const CASTLE_BONUS: i32 = 30;
const OFF_INITIAL_SQUARE_BONUS: i32 = 15;
const OPEN_FILE_BONUS: i32 = 21;

const INNER_KNIGHT_BONUS: i32 = 20;
const MIDDLE_KNIGHT_BONUS: i32 = OFF_INITIAL_SQUARE_BONUS;
const OUTER_KNIGHT_BONUS: i32 = -10;

// Relative values of single piece
fn value_of(piece: PieceEnum) -> i32 {
    match piece {
        PAWN => 100,
        KNIGHT => 300,
        BISHOP => 330,
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

    // note: knights not included here because they are incentived by the
    // active_knight_bonus
    const WHITE_INITIAL_SQUARES: [(Square, PieceEnum); 4] =
        [(A1, ROOK), (C1, BISHOP), (F1, BISHOP), (H1, ROOK)];
    const BLACK_INITIAL_SQUARES: [(Square, PieceEnum); 4] =
        [(A8, ROOK), (C8, BISHOP), (F8, BISHOP), (H8, ROOK)];

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

    white_bonus += active_knight_bonus(board.presence_white.knight);
    black_bonus -= active_knight_bonus(board.presence_black.knight);

    Ok(Score(
        white_value - black_value + white_bonus - black_bonus + castle_bonus,
    ))
}

// TODO: fix this, technically gives a bonus if the piece is captured
fn off_initial_square_bonus(
    board: &Board,
    color: Color,
    initial_squares: [(Square, PieceEnum); 4],
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

// knights get a bonus for being closer to the center of the board
fn active_knight_bonus(knights: Bitboard) -> i32 {
    // edge of the board
    let outer: Bitboard = A_FILE | H_FILE | RANK_1 | RANK_8;
    // next ring in
    let middle: Bitboard = (B_FILE | G_FILE | RANK_2 | RANK_7) & !outer;
    // center 4 ranks and files
    let inner: Bitboard = Bitboard::full() & !(outer | middle);

    (outer & knights).popcnt() as i32 * OUTER_KNIGHT_BONUS
        + (middle & knights).popcnt() as i32 * MIDDLE_KNIGHT_BONUS
        + (inner & knights).popcnt() as i32 * INNER_KNIGHT_BONUS
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

#[test]
fn test_active_knights() {
    let board = fen::parse("5n2/8/1n6/8/3N4/8/5N2/N7 w - - 0 1").unwrap();

    assert_eq!(
        active_knight_bonus(board.presence_white.knight),
        INNER_KNIGHT_BONUS + MIDDLE_KNIGHT_BONUS + OUTER_KNIGHT_BONUS
    );

    assert_eq!(
        active_knight_bonus(board.presence_black.knight),
        MIDDLE_KNIGHT_BONUS + OUTER_KNIGHT_BONUS
    );
}
