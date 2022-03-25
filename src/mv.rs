use crate::board::{square_symbol, Piece};
use crate::square::*;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Move {
    CastleKingside,
    CastleQueenside,
    Single {
        from: Square,
        to: Square,
    },
    Promote {
        from: Square,
        to: Square,
        piece: Piece, // TODO; only store PieceEnum
    },
}

impl Move {
    pub fn new(from: Square, to: Square) -> Self {
        Move::Single { from: from, to: to }
    }
}

impl From<(Square, Square)> for Move {
    fn from((from, to): (Square, Square)) -> Self {
        Move::Single { from, to }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Move::Single { from, to } => write!(f, "{}{}", from, to),
            Move::Promote { from, to, piece } => {
                write!(f, "{}{}{}", from, to, square_symbol(piece))
            }
            Move::CastleKingside => write!(f, "0-0"),
            Move::CastleQueenside => write!(f, "0-0-0"),
        }
    }
}

// struct MoveIterator<'a> {
//     board: &'a Board,
// }

// impl<'a> Iterator for MoveIterator<'a> {
//     type Item = Move;
//     fn next(&mut self) -> Option<Vec<&'a T>> { ... }
// }
