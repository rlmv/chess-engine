use core::cmp::Ordering;
use std::fmt;

/*
 * TODO: 0- or 1-indexing?
 */

const EMPTY: u8 = 0b00000000;
const PAWN: u8 = 0b00000001;
const KNIGHT: u8 = 0b00000010;
const BISHOP: u8 = 0b00000011;
const ROOK: u8 = 0b000000100;
const QUEEN: u8 = 0b00000101;
const KING: u8 = 0b00000110;

const PIECE_MASK: u8 = 0b00000111;

const BLACK: u8 = 0b01000000;
const WHITE: u8 = 0b10000000;

const COLOR_MASK: u8 = 0b11000000;

const N_RANKS: usize = 8;
const N_FILES: usize = 8;
const N_SQUARES: usize = N_RANKS * N_FILES;

#[derive(Debug)]
enum BoardError {
    NoPieceOnFromSquare(Square),
    NotImplemented,
}

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoardError::NoPieceOnFromSquare(square) => {
                write!(f, "Square {:?} does not have a piece", square)
            }
            BoardError::NotImplemented => {
                write!(f, "Missing implementation")
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Rank {
    _1,
    _2,
    _3,
    _4,
    _5,
    _6,
    _7,
    _8,
}

const RANKS: [Rank; 8] = [
    Rank::_1,
    Rank::_2,
    Rank::_3,
    Rank::_4,
    Rank::_5,
    Rank::_6,
    Rank::_7,
    Rank::_8,
];

#[derive(Debug, Eq, PartialEq)]
enum File {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}

const FILES: [File; 8] = [
    File::A,
    File::B,
    File::C,
    File::D,
    File::E,
    File::F,
    File::G,
    File::H,
];

#[derive(Clone)]
struct Board {
    board: [u8; 64],
}

impl Board {
    fn empty() -> Board {
        Board { board: [EMPTY; 64] }
    }

    /*
     * Convert a square, eg F2, to an index into the board array.
     */
    fn square_index(of: &Square) -> usize {
        let &Square(file, rank) = &of;
        Board::index(file, rank)
    }

    fn index(file: &File, rank: &Rank) -> usize {
        (Board::rank_index(rank) * 8 + Board::file_index(file)).into()
    }

    fn rank_index(rank: &Rank) -> u8 {
        use Rank::*;

        match rank {
            _1 => 7,
            _2 => 6,
            _3 => 5,
            _4 => 4,
            _5 => 3,
            _6 => 2,
            _7 => 1,
            _8 => 0,
        }
    }

    fn file_index(file: &File) -> u8 {
        use File::*;

        match file {
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

    fn place_piece(&self, piece: u8, on: Square) -> Board {
        let mut new = self.clone();
        new.board[Board::square_index(&on)] = piece;
        new
    }

    fn move_piece(&self, from: Square, to: Square) -> Result<Board, BoardError> {
        let mut new = self.clone();

        let i = Board::square_index(&from);
        let j = Board::square_index(&to);

        if new.board[i] & PIECE_MASK == EMPTY {
            return Err(BoardError::NoPieceOnFromSquare(from));
        }

        new.board[j] = new.board[i];
        new.board[i] = EMPTY;
        Ok(new)
    }

    // tODO: color
    fn is_occupied(&self, square: usize) -> bool {
        self.board[square] & PIECE_MASK != 0
    }

    fn is_empty(&self, square: usize) -> bool {
        self.board[square] & PIECE_MASK == 0
    }

    fn possible_moves(&self, from: Square) -> Result<Vec<Square>, BoardError> {
        // TODO: check color, turn
        match self.board[Board::square_index(&from)] & PIECE_MASK {
            EMPTY => Err(BoardError::NoPieceOnFromSquare(from)),
            ROOK => Ok(self._rook_moves(from)),
            _ => Err(BoardError::NotImplemented),
        }
    }

    fn _rook_moves(&self, from: Square) -> Vec<Square> {
        let index = Board::square_index(&from);

        // up-down
        let mut moves: Vec<Square> = (0..N_RANKS)
            .map(|j| index % N_RANKS + (j * N_FILES)) // enumerate from top of board
            .map(|k| Square::from_index(k))
            .filter(|s| *s != from)
            .collect();

        // side-to-side
        moves.extend(
            (0..N_FILES)
                .map(|j| (index / N_RANKS) * N_FILES + j) // enumerate from beginning of row
                .map(|k| Square::from_index(k))
                .filter(|s| *s != from),
        );

        let mut maybe_moves: Vec<Square> = Vec::new();

        // coordinates
        // let row = index / N_RANKS;
        // let col = index % N_RANKS;

        // side-to-side

        let mut target = index - 1;
        while target % N_FILES != (N_FILES - 1) {
            if self.is_occupied(target) {
                break;
            } else {
                maybe_moves.push(Square::from_index(target));
                target -= 1; // go back a file
            }
        }

        target = index + 1;
        while target % N_FILES != 0 {
            if self.is_occupied(target) {
                break;
            } else {
                maybe_moves.push(Square::from_index(target));
                target += 1; //advance a file
            }
        }

        target = index;
        while target >= N_FILES {
            // avoid underflow
            target -= N_FILES; // go back a row
            if self.is_occupied(target) {
                break;
            } else {
                maybe_moves.push(Square::from_index(target));
            }
        }

        target = index + N_FILES;
        while target < N_SQUARES {
            if self.is_occupied(target) {
                break;
            } else {
                maybe_moves.push(Square::from_index(target));
                target += N_FILES; // advance a row
            }
        }

        // for j in (1..=col).rev() {
        //     maybe_moves.push(Square::from_index(index - j));
        // }

        // for j in (col+1)..8 {
        //     maybe_moves.push(Square::from_index(index + j));
        // }

        // up-down

        // for i in (1..=row).rev() {
        //     maybe_moves.push(Square::from_index(index - ((i * N_FILES))));
        // }

        // }

        maybe_moves
    }
}

fn square_symbol(p: u8) -> char {
    let uncolored = match p & PIECE_MASK {
        EMPTY => ' ',
        PAWN => 'p',
        KNIGHT => 'n',
        BISHOP => 'b',
        ROOK => 'r',
        QUEEN => 'q',
        KING => 'k',
        unknown => panic!("Unknown piece {}", unknown),
    };

    match p & COLOR_MASK {
        BLACK => uncolored,
        WHITE => uncolored.to_ascii_uppercase(),
        unknown if unknown & PIECE_MASK == EMPTY => ' ',
        unknown => panic!("Unknown color {}, not possible", unknown),
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Square(File, Rank);

impl Square {
    fn new(file: File, rank: Rank) -> Square {
        Square(file, rank)
    }

    fn from_index(i: usize) -> Square {
        use File::*;
        use Rank::*;

        if i >= N_SQUARES {
            panic!("Square index {} is larger than max {}", i, N_SQUARES);
        }

        let rank = match i / N_RANKS {
            7 => _1,
            6 => _2,
            5 => _3,
            4 => _4,
            3 => _5,
            2 => _6,
            1 => _7,
            0 => _8,
            _ => panic!("Unknown rank"),
        };

        let file = match i % N_FILES {
            0 => A,
            1 => B,
            2 => C,
            3 => D,
            4 => E,
            5 => F,
            6 => G,
            7 => H,
            _ => panic!("Unknown file"),
        };

        Square::new(file, rank)
    }

    fn index(&self) -> usize {
        Board::square_index(self)
    }
}

impl From<(File, Rank)> for Square {
    fn from((file, rank): (File, Rank)) -> Self {
        Square::new(file, rank)
    }
}

impl Ord for Square {
    fn cmp(&self, other: &Self) -> Ordering {
        Board::square_index(self).cmp(&Board::square_index(other))
    }
}

impl PartialOrd for Square {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in RANKS.iter().rev() {
            write!(f, "{} ", Board::rank_index(rank) + 1)?;

            for file in FILES.iter() {
                let piece = self.board[Board::index(file, rank)];
                write!(f, "{} ", square_symbol(piece))?;
            }
            write!(f, "\n")?;
        }

        for c in vec![' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] {
            write!(f, "{} ", c)?;
        }

        fmt::Result::Ok(())
    }
}

fn main() {
    let b = Board::empty()
        .place_piece(PAWN | WHITE, Square::new(File::A, Rank::_2))
        .place_piece(KNIGHT | WHITE, Square::new(File::B, Rank::_1))
        .place_piece(BISHOP | WHITE, Square::new(File::C, Rank::_1))
        .place_piece(KING | WHITE, Square::new(File::E, Rank::_1))
        .place_piece(BISHOP | BLACK, Square::new(File::C, Rank::_8))
        .place_piece(KING | BLACK, Square::new(File::E, Rank::_8));

    // b.board[2] = KNIGHT | WHITE;
    // b.board[3] = BISHOP | WHITE;
    // b.board[50] = BISHOP | BLACK;
    // b.board[51] = PAWN | BLACK;

    println!("{}", b);
    println!(
        "{}",
        b.move_piece(
            Square::new(File::A, Rank::_2),
            Square::new(File::A, Rank::_4)
        )
        .unwrap()
    );

    // // println!(
    // //     "{}",
    // //     b.move_piece(Square { rank: 3, file: 2 }, Square { rank: 1, file: 2 })
    // //         .unwrap()
    // // );

    // println!("{}", PAWN);
    // println!("{}", KNIGHT);
    // println!("{}", BISHOP);
    // println!("{}", ROOK);
    // println!("{}", QUEEN);
    //    println!("{}", KING);

    println!("{}", square_symbol(QUEEN | WHITE));
}

fn sorted(mut v: Vec<Square>) -> Vec<Square> {
    v.sort();
    v
}

#[test]
fn test_rook_free_movement() {
    let board = Board::empty()
        .place_piece(ROOK | WHITE, Square(File::A, Rank::_1))
        .place_piece(ROOK | WHITE, Square(File::C, Rank::_6));

    assert_eq!(
        sorted(board.possible_moves(Square(File::A, Rank::_1)).unwrap()),
        sorted(vec![
            // rank moves (up-down)
            (File::A, Rank::_2).into(),
            (File::A, Rank::_3).into(),
            (File::A, Rank::_4).into(),
            (File::A, Rank::_5).into(),
            (File::A, Rank::_6).into(),
            (File::A, Rank::_7).into(),
            (File::A, Rank::_8).into(),
            // file moves (side-to-side)
            (File::B, Rank::_1).into(),
            (File::C, Rank::_1).into(),
            (File::D, Rank::_1).into(),
            (File::E, Rank::_1).into(),
            (File::F, Rank::_1).into(),
            (File::G, Rank::_1).into(),
            (File::H, Rank::_1).into(),
        ])
    );

    assert_eq!(
        sorted(board.possible_moves(Square(File::C, Rank::_6)).unwrap()),
        sorted(vec![
            // rank moves (up-down)
            (File::C, Rank::_1).into(),
            (File::C, Rank::_2).into(),
            (File::C, Rank::_3).into(),
            (File::C, Rank::_4).into(),
            (File::C, Rank::_5).into(),
            (File::C, Rank::_7).into(),
            (File::C, Rank::_8).into(),
            // file moves (side-to-side)
            (File::A, Rank::_6).into(),
            (File::B, Rank::_6).into(),
            (File::D, Rank::_6).into(),
            (File::E, Rank::_6).into(),
            (File::F, Rank::_6).into(),
            (File::G, Rank::_6).into(),
            (File::H, Rank::_6).into(),
        ])
    );
}

#[test]
fn test_rook_obstructed_movement() {
    let board = Board::empty()
        .place_piece(ROOK | WHITE, Square(File::E, Rank::_5))
        .place_piece(PAWN | WHITE, Square(File::E, Rank::_2))
        .place_piece(PAWN | WHITE, Square(File::E, Rank::_6))
        .place_piece(PAWN | WHITE, Square(File::A, Rank::_5))
        .place_piece(PAWN | WHITE, Square(File::G, Rank::_5));

    assert_eq!(
        sorted(board.possible_moves(Square(File::E, Rank::_5)).unwrap()),
        sorted(vec![
            // rank moves (up-down)
            (File::E, Rank::_3).into(),
            (File::E, Rank::_4).into(),
            // file moves (side-to-side)
            (File::B, Rank::_5).into(),
            (File::C, Rank::_5).into(),
            (File::D, Rank::_5).into(),
            (File::F, Rank::_5).into(),
        ])
    );
}

#[test]
fn test_is_empty() {
    let board = Board::empty()
        .place_piece(ROOK | WHITE, Square(File::A, Rank::_1))
        .place_piece(PAWN | BLACK, Square(File::C, Rank::_6));

    assert!(!board.is_empty(Square(File::A, Rank::_1).index()));
    assert!(board.is_occupied(Square(File::A, Rank::_1).index()));

    assert!(!board.is_empty(Square(File::C, Rank::_6).index()));
    assert!(board.is_occupied(Square(File::C, Rank::_6).index()));

    assert!(board.is_empty(Square(File::D, Rank::_3).index()));
    assert!(!board.is_occupied(Square(File::D, Rank::_3).index()));
}
