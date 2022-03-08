use std::fmt;

use crate::constants::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
pub enum File {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}

impl File {
    pub fn index(&self) -> u8 {
        use File::*;

        match self {
            A => 0,
            B => 1,
            C => 2,
            D => 3,
            E => 4,
            F => 5,
            G => 6,
            H => 7,
        }
    }

    pub fn from_index(i: usize) -> Self {
        use File::*;

        match i % N_FILES {
            0 => A,
            1 => B,
            2 => C,
            3 => D,
            4 => E,
            5 => F,
            6 => G,
            7 => H,
            _ => panic!("Unknown file"),
        }
    }

    pub fn from_str(s: &str) -> Self {
        use File::*;

        match &*s.to_ascii_uppercase() {
            "A" => A,
            "B" => B,
            "C" => C,
            "D" => D,
            "E" => E,
            "F" => F,
            "G" => G,
            "H" => H,
            _ => panic!("Unknown file"),
        }
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub const FILES: [File; 8] = [
    File::A,
    File::B,
    File::C,
    File::D,
    File::E,
    File::F,
    File::G,
    File::H,
];
