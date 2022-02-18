#![allow(dead_code)]

use core::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;

use crate::constants::*;
use crate::error::BoardError;
use crate::file::*;
use crate::rank::*;

type Result<T> = std::result::Result<T, BoardError>;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct Square(pub File, pub Rank);

impl Square {
    pub fn new(file: File, rank: Rank) -> Square {
        Square(file, rank)
    }

    pub fn rank(&self) -> &Rank {
        let Square(_, rank) = self;
        rank
    }

    pub fn file(&self) -> &File {
        let Square(file, _) = self;
        file
    }

    pub fn from_index(i: usize) -> Square {
        if i >= N_SQUARES {
            panic!("Square index {} is larger than max {}", i, N_SQUARES);
        }

        let rank = Rank::from_index(i);
        let file = File::from_index(i);

        Square::new(file, rank)
    }

    /*
     * Convert a square, eg F2, to an index into the board array.
     */
    pub fn index(&self) -> usize {
        let &Square(file, rank) = &self;
        (rank.index() as usize * N_FILES + file.index() as usize).into()
    }
}

pub const A1: Square = Square(File::A, Rank::_1);
pub const A2: Square = Square(File::A, Rank::_2);
pub const A3: Square = Square(File::A, Rank::_3);
pub const A4: Square = Square(File::A, Rank::_4);
pub const A5: Square = Square(File::A, Rank::_5);
pub const A6: Square = Square(File::A, Rank::_6);
pub const A7: Square = Square(File::A, Rank::_7);
pub const A8: Square = Square(File::A, Rank::_8);
pub const B1: Square = Square(File::B, Rank::_1);
pub const B2: Square = Square(File::B, Rank::_2);
pub const B3: Square = Square(File::B, Rank::_3);
pub const B4: Square = Square(File::B, Rank::_4);
pub const B5: Square = Square(File::B, Rank::_5);
pub const B6: Square = Square(File::B, Rank::_6);
pub const B7: Square = Square(File::B, Rank::_7);
pub const B8: Square = Square(File::B, Rank::_8);
pub const C1: Square = Square(File::C, Rank::_1);
pub const C2: Square = Square(File::C, Rank::_2);
pub const C3: Square = Square(File::C, Rank::_3);
pub const C4: Square = Square(File::C, Rank::_4);
pub const C5: Square = Square(File::C, Rank::_5);
pub const C6: Square = Square(File::C, Rank::_6);
pub const C7: Square = Square(File::C, Rank::_7);
pub const C8: Square = Square(File::C, Rank::_8);
pub const D1: Square = Square(File::D, Rank::_1);
pub const D2: Square = Square(File::D, Rank::_2);
pub const D3: Square = Square(File::D, Rank::_3);
pub const D4: Square = Square(File::D, Rank::_4);
pub const D5: Square = Square(File::D, Rank::_5);
pub const D6: Square = Square(File::D, Rank::_6);
pub const D7: Square = Square(File::D, Rank::_7);
pub const D8: Square = Square(File::D, Rank::_8);
pub const E1: Square = Square(File::E, Rank::_1);
pub const E2: Square = Square(File::E, Rank::_2);
pub const E3: Square = Square(File::E, Rank::_3);
pub const E4: Square = Square(File::E, Rank::_4);
pub const E5: Square = Square(File::E, Rank::_5);
pub const E6: Square = Square(File::E, Rank::_6);
pub const E7: Square = Square(File::E, Rank::_7);
pub const E8: Square = Square(File::E, Rank::_8);
pub const F1: Square = Square(File::F, Rank::_1);
pub const F2: Square = Square(File::F, Rank::_2);
pub const F3: Square = Square(File::F, Rank::_3);
pub const F4: Square = Square(File::F, Rank::_4);
pub const F5: Square = Square(File::F, Rank::_5);
pub const F6: Square = Square(File::F, Rank::_6);
pub const F7: Square = Square(File::F, Rank::_7);
pub const F8: Square = Square(File::F, Rank::_8);
pub const G1: Square = Square(File::G, Rank::_1);
pub const G2: Square = Square(File::G, Rank::_2);
pub const G3: Square = Square(File::G, Rank::_3);
pub const G4: Square = Square(File::G, Rank::_4);
pub const G5: Square = Square(File::G, Rank::_5);
pub const G6: Square = Square(File::G, Rank::_6);
pub const G7: Square = Square(File::G, Rank::_7);
pub const G8: Square = Square(File::G, Rank::_8);
pub const H1: Square = Square(File::H, Rank::_1);
pub const H2: Square = Square(File::H, Rank::_2);
pub const H3: Square = Square(File::H, Rank::_3);
pub const H4: Square = Square(File::H, Rank::_4);
pub const H5: Square = Square(File::H, Rank::_5);
pub const H6: Square = Square(File::H, Rank::_6);
pub const H7: Square = Square(File::H, Rank::_7);
pub const H8: Square = Square(File::H, Rank::_8);

impl From<(File, Rank)> for Square {
    fn from((file, rank): (File, Rank)) -> Self {
        Square::new(file, rank)
    }
}

impl TryFrom<&str> for Square {
    type Error = BoardError;

    fn try_from(s: &str) -> Result<Self> {
        crate::fen::parse_square(s)
    }
}

impl Ord for Square {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index().cmp(&other.index())
    }
}

impl PartialOrd for Square {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Square(file, rank) = self;
        write!(f, "{}{}", file, rank)
    }
}

#[test]
fn test_parse_square() {
    assert_eq!(Square::try_from("A1").unwrap(), Square(File::A, Rank::_1));
    assert_eq!(Square::try_from("B2").unwrap(), Square(File::B, Rank::_2));
    assert_eq!(Square::try_from("C3").unwrap(), Square(File::C, Rank::_3));
    assert_eq!(Square::try_from("D4").unwrap(), Square(File::D, Rank::_4));
    assert_eq!(Square::try_from("E5").unwrap(), Square(File::E, Rank::_5));
    assert_eq!(Square::try_from("F6").unwrap(), Square(File::F, Rank::_6));
    assert_eq!(Square::try_from("G7").unwrap(), Square(File::G, Rank::_7));
    assert_eq!(Square::try_from("H8").unwrap(), Square(File::H, Rank::_8));

    assert!(Square::try_from("I8").is_err());
}
