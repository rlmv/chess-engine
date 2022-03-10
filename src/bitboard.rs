use crate::color::*;
use crate::file::*;
use crate::rank::*;
use crate::square::*;
use std::fmt;
use std::mem;
use std::ops::{BitAnd, BitOr, Not};

/*
 * See https://rhysre.net/fast-chess-move-generation-with-magic-bitboards.html
 */

const A_FILE: Bitboard = Bitboard(0x101010101010101);
const H_FILE: Bitboard = Bitboard(0x8080808080808080);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Bitboard(u64);

impl Bitboard {
    pub fn empty() -> Self {
        Bitboard(0x0)
    }

    pub fn full() -> Self {
        Self(u64::MAX)
    }

    pub fn set(&self, square: Square) -> Self {
        Bitboard(self.0 | 1 << square.index())
    }

    pub fn set_all(squares: Vec<Square>) -> Self {
        squares
            .into_iter()
            .fold(Bitboard::empty(), |board, square| board.set(square))
    }

    /*
     * Return all set squares in the bitboard
     */
    pub fn squares<'a>(&self) -> SquareIterator {
        SquareIterator::new(self)
    }

    pub fn north_by(&self, ranks: u8) -> Self {
        Bitboard(self.0 << ranks * 8)
    }

    pub fn south_by(&self, ranks: u8) -> Self {
        Bitboard(self.0 >> ranks * 8)
    }

    pub fn east_by(&self, files: u8) -> Self {
        Bitboard(self.0 << files)
    }

    pub fn west_by(&self, files: u8) -> Self {
        Bitboard(self.0 >> files)
    }
}

pub struct SquareIterator<'a> {
    bitboard: &'a Bitboard,
    offset: u32, // next index in bitboard to inspect
}

impl<'a> SquareIterator<'a> {
    fn new(bitboard: &'a Bitboard) -> Self {
        SquareIterator {
            bitboard: bitboard,
            offset: 0,
        }
    }
}

impl<'a> Iterator for SquareIterator<'a> {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bitboard.0 >> self.offset == 0 {
            None
        } else {
            let next = (self.bitboard.0 >> self.offset).trailing_zeros();
            let square = Square::from_index((self.offset + next) as usize);
            self.offset += next + 1;

            Some(square)
        }
    }
}

impl BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl fmt::LowerHex for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitboard(")?;
        fmt::LowerHex::fmt(&self.0, f)?; // delegate to u64's implementation
        write!(f, ")")
    }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut bits: Vec<bool> = Vec::new();

        let Bitboard(board) = self;

        for i in 0..64 {
            let bit = (board >> i) & 0b1;
            bits.push(bit == 1);
        }

        for rank in RANKS.iter().rev() {
            for file in FILES.iter() {
                let index = Square::new(*file, *rank).index();
                let symbol = if bits[index] { '1' } else { '.' };
                write!(f, "{} ", symbol)?;
            }
            write!(f, "\n")?;
        }

        fmt::Result::Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PawnPresenceBitboard {
    b: Bitboard,
    color: Color,
}

impl PawnPresenceBitboard {
    pub fn empty(color: Color) -> Self {
        Self {
            b: Bitboard::empty(),
            color: color,
        }
    }

    pub fn set(&self, square: Square) -> Self {
        Self {
            b: self.b.set(square),
            color: self.color,
        }
    }

    pub fn attacks(&self) -> Bitboard {
        match self.color {
            WHITE => {
                let east_attacks = self.b.north_by(1).east_by(1) & !A_FILE;
                let west_attacks = self.b.north_by(1).west_by(1) & !H_FILE;
                east_attacks | west_attacks
            }
            BLACK => {
                let east_attacks = self.b.south_by(1).east_by(1) & !A_FILE;
                let west_attacks = self.b.south_by(1).west_by(1) & !H_FILE;
                east_attacks | west_attacks
            }
        }
    }
}

#[test]
fn test_pawn_attacks_white() {
    let pawn_presence = PawnPresenceBitboard {
        b: Bitboard::empty().set(A6).set(D2).set(H4),
        color: WHITE,
    };
    assert_eq!(
        pawn_presence.attacks(),
        Bitboard::empty().set(B7).set(C3).set(E3).set(G5)
    );
}

#[test]
fn test_pawn_attacks_black() {
    let pawn_presence = PawnPresenceBitboard {
        b: Bitboard::empty().set(A6).set(D2).set(H4),
        color: BLACK,
    };
    assert_eq!(
        pawn_presence.attacks(),
        Bitboard::empty().set(B5).set(C1).set(E1).set(G3)
    );
}

#[test]
fn test_square() {
    let b = Bitboard::set_all(vec![D2, H4, A6]);
    assert_eq!(b.squares().collect::<Vec<Square>>(), vec![D2, H4, A6]);
}

#[test]
fn test_print() {
    let board = !H_FILE;
    println!("{}", board);
    println!("{:?}", board);
    println!("{:#x}", board);
    assert!(false);
}
