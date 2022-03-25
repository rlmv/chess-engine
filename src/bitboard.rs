use crate::board::*;
use crate::color::*;
use crate::file::*;
use crate::rank::*;
use crate::square::*;
use crate::vector::*;
use std::fmt;
use std::ops::{BitAnd, BitOr, BitXor, BitXorAssign, Not, Shl, Shr};

use lazy_static::lazy_static;

/*
 * See https://rhysre.net/fast-chess-move-generation-with-magic-bitboards.html
 */

// TODO: any runtime costs associated with lazy_static?
lazy_static! {
    pub static ref PRECOMPUTED_BITBOARDS: PrecomputedBitboards = precompute_bitboards();
}

pub struct PrecomputedBitboards {
    pub king_moves: [Bitboard; 64], // TODO: can we index this by square directly?
    pub knight_moves: [Bitboard; 64],
    pub rays: Rays, // for sliding pieces
}

pub struct Rays {
    pub north: [Bitboard; 64],
    pub north_east: [Bitboard; 64],
    pub east: [Bitboard; 64],
    pub south_east: [Bitboard; 64],
    pub south: [Bitboard; 64],
    pub south_west: [Bitboard; 64],
    pub west: [Bitboard; 64],
    pub north_west: [Bitboard; 64],
}

fn precompute_bitboards() -> PrecomputedBitboards {
    PrecomputedBitboards {
        king_moves: king_moves(),
        knight_moves: knight_moves(),
        rays: rays(),
    }
}

fn king_moves() -> [Bitboard; 64] {
    let mut king_moves = [Bitboard::empty(); 64];

    const MOVE_VECTORS: [MoveVector; 8] = [
        MoveVector(1, 1),
        MoveVector(1, 0),
        MoveVector(1, -1),
        MoveVector(0, -1),
        MoveVector(-1, -1),
        MoveVector(-1, 0),
        MoveVector(-1, 1),
        MoveVector(0, 1),
    ];

    for from in Square::all_squares() {
        for target in MOVE_VECTORS
            .iter()
            .filter_map(|v| Board::plus_vector(&from, v))
        {
            king_moves[from.index()] = king_moves[from.index()].set(target)
        }
    }

    king_moves
}

fn knight_moves() -> [Bitboard; 64] {
    let mut knight_moves = [Bitboard::empty(); 64];

    const MOVE_VECTORS: [MoveVector; 8] = [
        MoveVector(1, 2),
        MoveVector(2, 1),
        MoveVector(2, -1),
        MoveVector(1, -2),
        MoveVector(-1, -2),
        MoveVector(-2, -1),
        MoveVector(-2, 1),
        MoveVector(-1, 2),
    ];

    for from in Square::all_squares() {
        for target in MOVE_VECTORS
            .iter()
            .filter_map(|v| Board::plus_vector(&from, v))
        {
            knight_moves[from.index()] = knight_moves[from.index()].set(target)
        }
    }

    knight_moves
}

fn rays() -> Rays {
    const MAX_MAGNITUDE: u8 = 7;

    fn compute_rays(v: MoveVector) -> [Bitboard; 64] {
        let mut rays = [Bitboard::empty(); 64];

        for from in Square::all_squares() {
            // Iterate allowed vectors, scaling by all possible magnitudes
            for target in Board::plus_vector_scaled(&from, &v, MAX_MAGNITUDE) {
                rays[from.index()] = rays[from.index()].set(target);
            }
        }

        rays
    }

    Rays {
        north: compute_rays(MoveVector(0, 1)),
        north_east: compute_rays(MoveVector(1, 1)),
        east: compute_rays(MoveVector(1, 0)),
        south_east: compute_rays(MoveVector(1, -1)),
        south: compute_rays(MoveVector(0, -1)),
        south_west: compute_rays(MoveVector(-1, -1)),
        west: compute_rays(MoveVector(-1, 0)),
        north_west: compute_rays(MoveVector(-1, 1)),
    }
}

pub const A_FILE: Bitboard = Bitboard(0x101010101010101);
pub const H_FILE: Bitboard = Bitboard(0x8080808080808080);

pub const RANK_1: Bitboard = Bitboard(0xff);
pub const RANK_8: Bitboard = Bitboard(0xff00000000000000);

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

    pub fn unset(&self, square: Square) -> Self {
        Bitboard(self.0 ^ 1 << square.index())
    }

    pub fn set_all(squares: &Vec<Square>) -> Self {
        squares
            .into_iter()
            .fold(Bitboard::empty(), |board, square| board.set(*square))
    }

    /*
     * Return all set squares in the bitboard
     */
    pub fn squares(&self) -> SquareIterator {
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

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn non_empty(&self) -> bool {
        self.0 != 0
    }

    // least significant set bit
    pub fn bitscan_forward(&self) -> Option<Square> {
        if self.is_empty() {
            None
        } else {
            Some(Square::from_index(self.0.trailing_zeros() as usize))
        }
    }

    // most significant set bit
    pub fn bitscan_backward(&self) -> Option<Square> {
        if self.is_empty() {
            None
        } else {
            Some(Square::from_index(63 - self.0.leading_zeros() as usize))
        }
    }
}

pub struct SquareIterator {
    bitboard: Bitboard,
    offset: u32, // next index in bitboard to inspect
}

impl SquareIterator {
    fn new(bitboard: &Bitboard) -> Self {
        SquareIterator {
            bitboard: *bitboard,
            offset: 0,
        }
    }
}

impl<'a> Iterator for SquareIterator {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        // will panic if tries to shift more than 64 bits
        if self.offset >= 64 || self.bitboard.0 >> self.offset == 0 {
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

impl BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Shl<usize> for Bitboard {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        Self(self.0 << rhs)
    }
}

impl Shr<usize> for Bitboard {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        Self(self.0 >> rhs)
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
    pub b: Bitboard,
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
    let b = Bitboard::set_all(&vec![D2, H4, A6]);
    assert_eq!(b.squares().collect::<Vec<Square>>(), vec![D2, H4, A6]);
}

#[test]
fn test_print() {
    let board = Bitboard::set_all(&vec![A8, B8, C8, D8, E8, F8, G8, H8]);
    println!("{}", board);
    println!("{:?}", board);
    println!("{:#x}", board);
    //    assert!(false);
}

#[test]
fn test_bitboard_squares_no_panic_when_h8_is_set() {
    let squares = vec![F8, H8];
    let bitboard = Bitboard::set_all(&squares);
    assert_eq!(squares, bitboard.squares().collect::<Vec<Square>>());
}

#[test]
fn test_precompute_bitboards() {
    for (i, bitboard) in PRECOMPUTED_BITBOARDS.rays.south.iter().enumerate() {
        println!("{}", Square::from_index(i));
        println!("{}", bitboard);
    }

    // assert!(false);
}
