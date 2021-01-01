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
