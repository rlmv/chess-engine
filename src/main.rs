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

#[derive(Debug)]
enum BoardError {
    NoPieceOnFromSquare(Square),
}

#[derive(Debug)]
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

#[derive(Debug)]
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

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoardError::NoPieceOnFromSquare(square) => {
                write!(f, "Square {:?} does not have a piece", square)
            }
        }
    }
}

#[derive(Clone)]
struct Board {
    board: [u8; 64],

    
}

impl Board {
    /*
     * Convert a square, eg F2, to an index into the board array.
     */
    fn square_index(of: &Square) -> usize {
        Board::index(&of.file, &of.rank)
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

#[derive(Debug)]
struct Square {
    file: File,
    rank: Rank,
}

impl Square {
    fn new(file: File, rank: Rank) -> Square {
        Square {
            file: file,
            rank: rank,
        }
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
    println!("Hello, world!");

    let board: [u8; 64] = [0; 64];
    //    println!("{:?}", board);

    let mut b = Board { board: board }
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
        b.move_piece(Square::new(File::A, Rank::_2), Square::new(File::A, Rank::_4))
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

    println!("{}", square_symbol(QUEEN |  WHITE));
}
