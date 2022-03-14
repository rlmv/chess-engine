#![allow(dead_code)]

use core::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;

use crate::constants::*;
use crate::error::BoardError;
use crate::file::*;
use crate::rank::*;

type Result<T> = std::result::Result<T, BoardError>;

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
pub struct Square(usize);

impl Square {
    pub fn new(file: File, rank: Rank) -> Square {
        Square::from_index((rank.index() * 8 + file.index()) as usize)
    }

    pub fn rank(&self) -> Rank {
        Rank::from_index(self.index())
    }

    pub fn file(&self) -> File {
        File::from_index(self.index())
    }

    pub fn from_index(i: usize) -> Square {
        //        TODO: any way to get rid of this bounds check for speed up?
        if i >= N_SQUARES {
            panic!("Square index {} is larger than max {}", i, N_SQUARES);
        }
        Square(i)
    }

    /*
     * Convert a square, eg F2, to an index into the board array.
     */
    pub fn index(&self) -> usize {
        self.0
    }

    pub fn all_squares() -> impl Iterator<Item = Square> {
        (0..64).map(|i| Square::from_index(i))
    }
}

pub const A1: Square = Square(0);
pub const B1: Square = Square(1);
pub const C1: Square = Square(2);
pub const D1: Square = Square(3);
pub const E1: Square = Square(4);
pub const F1: Square = Square(5);
pub const G1: Square = Square(6);
pub const H1: Square = Square(7);
pub const A2: Square = Square(8);
pub const B2: Square = Square(9);
pub const C2: Square = Square(10);
pub const D2: Square = Square(11);
pub const E2: Square = Square(12);
pub const F2: Square = Square(13);
pub const G2: Square = Square(14);
pub const H2: Square = Square(15);
pub const A3: Square = Square(16);
pub const B3: Square = Square(17);
pub const C3: Square = Square(18);
pub const D3: Square = Square(19);
pub const E3: Square = Square(20);
pub const F3: Square = Square(21);
pub const G3: Square = Square(22);
pub const H3: Square = Square(23);
pub const A4: Square = Square(24);
pub const B4: Square = Square(25);
pub const C4: Square = Square(26);
pub const D4: Square = Square(27);
pub const E4: Square = Square(28);
pub const F4: Square = Square(29);
pub const G4: Square = Square(30);
pub const H4: Square = Square(31);
pub const A5: Square = Square(32);
pub const B5: Square = Square(33);
pub const C5: Square = Square(34);
pub const D5: Square = Square(35);
pub const E5: Square = Square(36);
pub const F5: Square = Square(37);
pub const G5: Square = Square(38);
pub const H5: Square = Square(39);
pub const A6: Square = Square(40);
pub const B6: Square = Square(41);
pub const C6: Square = Square(42);
pub const D6: Square = Square(43);
pub const E6: Square = Square(44);
pub const F6: Square = Square(45);
pub const G6: Square = Square(46);
pub const H6: Square = Square(47);
pub const A7: Square = Square(48);
pub const B7: Square = Square(49);
pub const C7: Square = Square(50);
pub const D7: Square = Square(51);
pub const E7: Square = Square(52);
pub const F7: Square = Square(53);
pub const G7: Square = Square(54);
pub const H7: Square = Square(55);
pub const A8: Square = Square(56);
pub const B8: Square = Square(57);
pub const C8: Square = Square(58);
pub const D8: Square = Square(59);
pub const E8: Square = Square(60);
pub const F8: Square = Square(61);
pub const G8: Square = Square(62);
pub const H8: Square = Square(63);

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
        write!(f, "{}{}", self.file(), self.rank())
    }
}

#[test]
fn test_parse_square() {
    assert_eq!(Square::try_from("A1").unwrap(), A1);
    assert_eq!(Square::try_from("B2").unwrap(), B2);
    assert_eq!(Square::try_from("C3").unwrap(), C3);
    assert_eq!(Square::try_from("D4").unwrap(), D4);
    assert_eq!(Square::try_from("E5").unwrap(), E5);
    assert_eq!(Square::try_from("F6").unwrap(), F6);
    assert_eq!(Square::try_from("G7").unwrap(), G7);
    assert_eq!(Square::try_from("H8").unwrap(), H8);

    assert!(Square::try_from("I8").is_err());
}
