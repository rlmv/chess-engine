use crate::board::*;
use crate::color::*;
use crate::square::*;

use lazy_static::lazy_static;

type ZobristHash = u64;

/*
 * Compute the Zobrist hash for a board from scratch
 */
pub fn compute_hash(board: &Board) -> ZobristHash {
    let mut hash: ZobristHash = 0;

    for (piece, square) in board.all_pieces_of_color(WHITE) {
        hash ^= constant_for_piece(piece, square)
    }

    for (piece, square) in board.all_pieces_of_color(BLACK) {
        hash ^= constant_for_piece(piece, square)
    }

    if board.color_to_move == WHITE {
        hash ^= *ZOBRIST_WHITE_TO_MOVE;
    }

    // TODO: include castle rights
    // TODO: include move clocks?

    hash
}

pub struct ZobristColor {
    pawn: [u64; 64],
    bishop: [u64; 64],
    knight: [u64; 64],
    rook: [u64; 64],
    queen: [u64; 64],
    king: [u64; 64],
}

lazy_static! {
    pub static ref ZOBRIST_WHITE: ZobristColor = init_constants();
    pub static ref ZOBRIST_BLACK: ZobristColor = init_constants();
    pub static ref ZOBRIST_WHITE_TO_MOVE: u64 = rand::random::<u64>();
}

fn rand_array() -> [u64; 64] {
    [(); 64].map(|_| rand::random::<u64>())
}

fn init_constants() -> ZobristColor {
    ZobristColor {
        pawn: rand_array(),
        bishop: rand_array(),
        knight: rand_array(),
        rook: rand_array(),
        queen: rand_array(),
        king: rand_array(),
    }
}

fn constant_for_piece(piece: Piece, square: Square) -> u64 {
    let for_color: &ZobristColor = match piece.color() {
        WHITE => &ZOBRIST_WHITE,
        BLACK => &ZOBRIST_BLACK,
    };

    let for_piece: &[u64; 64] = match piece.piece() {
        PAWN => &for_color.pawn,
        KNIGHT => &for_color.knight,
        BISHOP => &for_color.bishop,
        ROOK => &for_color.rook,
        QUEEN => &for_color.queen,
        KING => &for_color.king,
    };

    for_piece[square.index()]
}

#[test]
fn test_zobrist_init() {
    dbg!(ZOBRIST_WHITE.pawn);

    let board = crate::fen::parse("5rk1/3n2pp/2p1p3/5pP1/1PPP4/8/5PP1/R5K1 b - - 0 26").unwrap();

    dbg!(compute_hash(&board));
    assert!(false);
}
