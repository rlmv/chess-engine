use crate::board::*;
use crate::color::*;
use crate::square::*;
use std::default::Default;
use std::hash::Hasher;
use std::ops::BitXorAssign;

use lazy_static::lazy_static;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ZobristHash(u64);

impl ZobristHash {
    pub fn new() -> Self {
        ZobristHash(0)
    }

    #[inline]
    pub fn toggle_piece(&mut self, piece: Piece, square: Square) -> &mut Self {
        *self ^= constant_for_piece(piece, square);
        self
    }

    #[inline]
    pub fn toggle_color_to_move(&mut self) -> &mut Self {
        *self ^= *ZOBRIST_WHITE_TO_MOVE;
        self
    }

    #[inline]
    pub fn toggle_en_passant_target(&mut self, square: Square) -> &mut Self {
        *self ^= ZOBRIST_EN_PASSANT_TARGET[square.index()];
        self
    }

    // TODO: can this be more efficient?
    pub fn toggle_castle_rights(&mut self, can_castle: CastleRights) -> &mut Self {
        if can_castle.kingside_white {
            *self ^= *ZOBRIST_CASTLE_KINGSIDE_WHITE
        }
        if can_castle.queenside_white {
            *self ^= *ZOBRIST_CASTLE_QUEENSIDE_WHITE
        }
        if can_castle.kingside_black {
            *self ^= *ZOBRIST_CASTLE_KINGSIDE_BLACK
        }
        if can_castle.queenside_black {
            *self ^= *ZOBRIST_CASTLE_QUEENSIDE_BLACK
        }

        self
    }
}

impl BitXorAssign for ZobristHash {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

/*
 * Compute the Zobrist hash for a board from scratch
 */
pub fn compute_hash(board: &Board) -> ZobristHash {
    let mut hash: ZobristHash = ZobristHash(0);

    for (piece, square) in board.all_pieces_of_color(WHITE) {
        hash ^= constant_for_piece(piece, square);
    }

    for (piece, square) in board.all_pieces_of_color(BLACK) {
        hash ^= constant_for_piece(piece, square);
    }

    if board.color_to_move == WHITE {
        hash ^= *ZOBRIST_WHITE_TO_MOVE;
    }

    if board.can_castle.kingside_white {
        hash ^= *ZOBRIST_CASTLE_KINGSIDE_WHITE
    }

    if board.can_castle.kingside_black {
        hash ^= *ZOBRIST_CASTLE_KINGSIDE_BLACK
    }

    if board.can_castle.queenside_white {
        hash ^= *ZOBRIST_CASTLE_QUEENSIDE_WHITE
    }

    if board.can_castle.queenside_black {
        hash ^= *ZOBRIST_CASTLE_QUEENSIDE_BLACK
    }

    if let Some(en_passant_target) = board.en_passant_target {
        hash ^= ZOBRIST_EN_PASSANT_TARGET[en_passant_target.index()]
    }

    // TODO: include en passant target
    // TODO: include move clocks?

    hash
}

pub struct ZobristColor {
    pawn: [ZobristHash; 64],
    bishop: [ZobristHash; 64],
    knight: [ZobristHash; 64],
    rook: [ZobristHash; 64],
    queen: [ZobristHash; 64],
    king: [ZobristHash; 64],
}

lazy_static! {
    pub static ref ZOBRIST_WHITE: ZobristColor = rand_color();
    pub static ref ZOBRIST_BLACK: ZobristColor = rand_color();
    pub static ref ZOBRIST_WHITE_TO_MOVE: ZobristHash = rand_hash();
    pub static ref ZOBRIST_CASTLE_KINGSIDE_WHITE: ZobristHash = rand_hash();
    pub static ref ZOBRIST_CASTLE_KINGSIDE_BLACK: ZobristHash = rand_hash();
    pub static ref ZOBRIST_CASTLE_QUEENSIDE_WHITE: ZobristHash = rand_hash();
    pub static ref ZOBRIST_CASTLE_QUEENSIDE_BLACK: ZobristHash = rand_hash();
    pub static ref ZOBRIST_EN_PASSANT_TARGET: [ZobristHash; 64] = rand_array();
}

fn rand_hash() -> ZobristHash {
    ZobristHash(rand::random::<u64>())
}

fn rand_array() -> [ZobristHash; 64] {
    [(); 64].map(|_| rand_hash())
}

fn rand_color() -> ZobristColor {
    ZobristColor {
        pawn: rand_array(),
        bishop: rand_array(),
        knight: rand_array(),
        rook: rand_array(),
        queen: rand_array(),
        king: rand_array(),
    }
}

fn constant_for_piece(piece: Piece, square: Square) -> ZobristHash {
    let for_color: &ZobristColor = match piece.color() {
        WHITE => &ZOBRIST_WHITE,
        BLACK => &ZOBRIST_BLACK,
    };

    let for_piece: &[ZobristHash; 64] = match piece.piece() {
        PAWN => &for_color.pawn,
        KNIGHT => &for_color.knight,
        BISHOP => &for_color.bishop,
        ROOK => &for_color.rook,
        QUEEN => &for_color.queen,
        KING => &for_color.king,
    };

    for_piece[square.index()]
}

/*
 * Custom hash implementation for Zobrist keys.
 *
 * Since the Zobrist key is already a hash, just use it directly instead of
 * rehashing the value.
 */
pub struct ZobristHasher {
    hash: u64,
}

impl Hasher for ZobristHasher {
    // The hasher should only be used to write u64s. Hard error if used for
    // another type.
    #[inline]
    fn write(&mut self, _: &[u8]) {
        panic!("not implemented")
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        // do not expect to write to this hasher more than once
        debug_assert_eq!(self.hash, 0);
        self.hash = i
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
impl Default for ZobristHasher {
    fn default() -> Self {
        Self { hash: 0 }
    }
}

#[test]
fn test_zobrist_init() {
    dbg!(ZOBRIST_WHITE.pawn);

    let board = crate::fen::parse("5rk1/3n2pp/2p1p3/5pP1/1PPP4/8/5PP1/R5K1 b - - 0 26").unwrap();

    dbg!(compute_hash(&board));
    //    assert!(false);
}
