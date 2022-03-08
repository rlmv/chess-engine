use std::fmt;

use crate::constants::*;

pub use Color::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Color {
    WHITE,
    BLACK,
}

impl Color {
    pub fn encode(&self) -> u8 {
        match self {
            WHITE => WHITE_BIT,
            BLACK => BLACK_BIT,
        }
    }

    pub fn opposite(&self) -> Color {
        match self {
            WHITE => BLACK,
            BLACK => WHITE,
        }
    }
}

impl From<u8> for Color {
    fn from(x: u8) -> Self {
        match x & COLOR_MASK {
            WHITE_BIT => WHITE,
            BLACK_BIT => BLACK,
            y => panic!("Unknown color {} for piece {}", y, x),
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
