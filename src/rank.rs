use crate::constants::*;
use std::fmt;

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
pub enum Rank {
    _1,
    _2,
    _3,
    _4,
    _5,
    _6,
    _7,
    _8,
}

impl Rank {
    pub fn index(&self) -> u8 {
        use Rank::*;
        match self {
            _1 => 0,
            _2 => 1,
            _3 => 2,
            _4 => 3,
            _5 => 4,
            _6 => 5,
            _7 => 6,
            _8 => 7,
        }
    }

    pub fn from_index(i: usize) -> Self {
        use Rank::*;
        match i / N_RANKS {
            0 => _1,
            1 => _2,
            2 => _3,
            3 => _4,
            4 => _5,
            5 => _6,
            6 => _7,
            7 => _8,
            _ => panic!("Unknown rank"),
        }
    }

    pub fn from_str(s: &str) -> Self {
        use Rank::*;
        match s {
            "1" => _1,
            "2" => _2,
            "3" => _3,
            "4" => _4,
            "5" => _5,
            "6" => _6,
            "7" => _7,
            "8" => _8,
            _ => panic!("Unknown rank"),
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).trim_start_matches("_"))
    }
}

pub const RANKS: [Rank; 8] = [
    Rank::_1,
    Rank::_2,
    Rank::_3,
    Rank::_4,
    Rank::_5,
    Rank::_6,
    Rank::_7,
    Rank::_8,
];
