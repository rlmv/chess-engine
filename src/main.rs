use core::cmp::Ordering;
use rand::random;
use regex::Regex;
use std::convert::TryFrom;
use std::fmt;
/*
 * TODO:
 *
 * - min-max optimization
 * - implement other pieces
 */

#[derive(Debug)]
struct Piece(u8, Color);

impl Piece {
    fn piece(&self) -> u8 {
        let Piece(piece, _) = self;
        piece & PIECE_MASK
    }

    fn color(&self) -> Color {
        let Piece(_, color) = self;
        *color
    }
    fn encode(&self) -> u8 {
        let Piece(piece, color) = *self;
        (piece & PIECE_MASK) | (color.encode())
    }

    fn from(x: u8) -> Option<Self> {
        if x & PIECE_MASK == EMPTY {
            return None;
        }

        let color = Color::from(x);

        Some(Piece(x & PIECE_MASK, color))
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Color {
    WHITE,
    BLACK,
}
use Color::*;

impl Color {
    fn encode(&self) -> u8 {
        match self {
            WHITE => WHITE_BIT,
            BLACK => BLACK_BIT,
        }
    }

    fn opposite(&self) -> Color {
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

const EMPTY: u8 = 0b00000000;
const PAWN: u8 = 0b00000001;
const KNIGHT: u8 = 0b00000010;
const BISHOP: u8 = 0b00000011;
const ROOK: u8 = 0b000000100;
const QUEEN: u8 = 0b00000101;
const KING: u8 = 0b00000110;

const PIECE_MASK: u8 = 0b00000111;

const BLACK_BIT: u8 = 0b01000000;
const WHITE_BIT: u8 = 0b10000000;
const COLOR_MASK: u8 = 0b11000000;

const N_RANKS: usize = 8;
const N_FILES: usize = 8;
const N_SQUARES: usize = N_RANKS * N_FILES;

#[derive(Debug)]
enum BoardError {
    NoPieceOnFromSquare(Square),
    NotImplemented,
    IllegalState(String),
    ParseError(String),
}
use BoardError::*;

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoPieceOnFromSquare(square) => {
                write!(f, "Square {:?} does not have a piece", square)
            }
            NotImplemented => write!(f, "Missing implementation"),
            IllegalState(msg) => write!(f, "{}", msg),
            ParseError(msg) => write!(f, "{}", msg),
        }
    }
}

impl From<regex::Error> for BoardError {
    // TODO: chain errors
    fn from(re: regex::Error) -> Self {
        BoardError::ParseError(format!("{:?}", re))
    }
}

type MoveVector = (i8, i8); // x, y

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
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

impl Rank {
    fn index(&self) -> u8 {
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

    fn from_index(i: usize) -> Self {
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

    fn from_str(s: &str) -> Self {
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

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
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

impl File {
    fn index(&self) -> u8 {
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

    fn from_index(i: usize) -> Self {
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

    fn from_str(s: &str) -> Self {
        use File::*;

        match s {
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

#[derive(Copy, Clone)]
struct Board {
    board: [u8; 64],
    color_to_move: Color,
}

impl Board {
    fn empty() -> Self {
        Board {
            board: [EMPTY; 64],
            color_to_move: WHITE,
        }
    }

    fn with_color_to_move(&self, color: Color) -> Self {
        let mut new = self.clone();
        new.color_to_move = color;
        new
    }

    /*
     * Convert a square, eg F2, to an index into the board array.
     */
    fn square_index(of: &Square) -> usize {
        let &Square(file, rank) = &of;
        (rank.index() * 8 + file.index()).into()
    }

    fn place_piece(&self, piece: Piece, on: Square) -> Board {
        let mut new = self.clone();

        new.board[Board::square_index(&on)] = piece.encode();
        new
    }

    fn move_piece(&self, from: Square, to: Square) -> Result<Board, BoardError> {
        let mut new = self.clone();

        let i = Board::square_index(&from);
        let j = Board::square_index(&to);

        if new.board[i] & PIECE_MASK == EMPTY {
            return Err(NoPieceOnFromSquare(from));
        }

        new.board[j] = new.board[i];
        new.board[i] = EMPTY;
        Ok(new)
    }

    fn is_occupied(&self, square: usize) -> bool {
        Piece::from(self.board[square]).is_some()
    }

    fn is_empty(&self, square: usize) -> bool {
        Piece::from(self.board[square]).is_none()
    }

    fn can_capture(&self, target: usize, attacker: Color) -> bool {
        self.is_occupied_by_color(target, attacker.opposite())
    }

    fn is_occupied_by_color(&self, square: usize, color: Color) -> bool {
        match Piece::from(self.board[square]) {
            Some(piece) if piece.color() == color => true,
            _ => false,
        }
    }

    fn is_in_check(&self, color: Color) -> Result<bool, BoardError> {
        let all_pieces = self.all_pieces_of_color(color);

        let kings: Vec<&(Piece, Square)> = all_pieces
            .iter()
            .filter(|(p, _)| p.piece() & PIECE_MASK == KING && p.color() == color)
            .collect();

        if kings.len() == 0 {
            return Err(IllegalState(format!(
                "Board is missing KING of color {}",
                color
            )));
        } else if kings.len() > 1 {
            return Err(IllegalState(format!(
                "Board has {} KINGs of color {}",
                kings.len(),
                color
            )));
        }

        let (king, king_square) = &kings[0];

        self.attacked_by_color(king_square.index(), king.color().opposite())
    }

    // attacking moves is a subset of other moves -
    fn attacked_by_color(&self, square: usize, color: Color) -> Result<bool, BoardError> {
        for (i, _) in self.board.iter().enumerate() {
            if self.is_occupied_by_color(i, color)
                && self
                    .possible_moves(Square::from_index(i), true)?
                    .contains(&Square::from_index(square))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn possible_moves(
        &self,
        from: Square,
        ignore_king_jeopardy: bool,
    ) -> Result<Vec<Square>, BoardError> {
        // TODO: check color, turn

        match Piece::from(self.board[Board::square_index(&from)]) {
            None => Err(NoPieceOnFromSquare(from)),
            Some(p) if p.piece() == ROOK => Ok(self._rook_moves(from, p)),
            Some(p) if p.piece() == KING => self._king_moves(from, p, ignore_king_jeopardy),
            _ => Err(NotImplemented),
        }
    }

    // TODO: this "ignore_king_jeopardy" flag feels hacky. Is there a way to
    // simplify and avoid this? More efficient way to find all attackers of a square?
    //
    // Maybe don't need to exlude moves into check? Can instead remove those later with the is_in_check option?
    fn _king_moves(
        &self,
        from: Square,
        king: Piece,
        ignore_king_jeopardy: bool,
    ) -> Result<Vec<Square>, BoardError> {
        let index = Board::square_index(&from);
        let mut moves: Vec<Square> = Vec::new();
        let move_vectors: Vec<MoveVector> = vec![
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
        ];

        let signed_index = index as i8;

        for (x, y) in move_vectors {
            let target = signed_index + x + (y * N_FILES as i8);

            if x == -1 && target % N_FILES as i8 == (N_FILES as i8 - 1) {
                // ignore: wrap around to left
            } else if x == 1 && target % N_FILES as i8 == 0 {
                // ignore: wrap around to right
            } else if target < 0 {
                // out bottom of board
            } else if target >= N_SQUARES as i8 {
                // out top
            } else if self.is_occupied_by_color(target as usize, king.color()) { // TODO usize cast is bad
                 // cannot move into square of own piece
            } else if !ignore_king_jeopardy
                && self.attacked_by_color(target as usize, king.color().opposite())?
            {
                // TODO usize cast is bad

                // Cannot move *into* check.
                // BUT: we don't care we are testing when the opponents king
                // moves into check. Don't recurse.
            } else {
                moves.push(Square::from_index(target as usize));
            }
        }

        Ok(moves)
    }

    fn _rook_moves(&self, from: Square, attacker: Piece) -> Vec<Square> {
        let index = Board::square_index(&from);

        let mut maybe_moves: Vec<Square> = Vec::new();

        let move_vectors: [MoveVector; 4] = [(1, 0), (0, -1), (-1, 0), (0, 1)];

        const MAX_MAGNITUDE: u8 = 7;

        let signed_index = index as i8;

        // Iterate allowed vectors, scaling by all possible magnitudes
        for (x, y) in move_vectors.iter() {
            for m in 1..=MAX_MAGNITUDE {
                let target = signed_index + (m as i8 * x) + (m as i8 * y * N_FILES as i8);

                if target >= N_SQUARES as i8 {
                    // off top of board
                    break;
                } else if target < 0 {
                    // out bottom of board
                    break;
                } else if *x == -1 && target % N_FILES as i8 == N_FILES as i8 - 1 {
                    // wrap around to left
                    break;
                } else if *x == 1 && target % N_FILES as i8 == 0 {
                    // wrap to right
                    break;
                } else if self.is_occupied_by_color(target as usize, attacker.color()) {
                    // TODO: remove usize cast
                    break;
                } else if self.can_capture(target as usize, attacker.color()) {
                    // TODO: remove usize cast
                    maybe_moves.push(Square::from_index(target as usize));
                    break;
                } else {
                    maybe_moves.push(Square::from_index(target as usize));
                }
            }
        }

        maybe_moves
    }

    fn all_pieces_of_color(&self, color: Color) -> Vec<(Piece, Square)> {
        let mut pieces: Vec<(Piece, Square)> = Vec::new();

        for (i, p) in self.board.iter().enumerate() {
            match Piece::from(*p) {
                Some(piece) if piece.color() == color => {
                    pieces.push((piece, Square::from_index(i)))
                }
                _ => (),
            }
        }

        pieces
    }

    fn find_next_move(&self) -> Result<Option<Move>, BoardError> {
        // Find all pieces
        // Generate all valid moves for those pieces.
        // After each move, must not be in check - prune.
        // Make each move - evaluate the position.
        // Pick highest scoring move

        // Evaluate checkmate?
        // Evaluate stalemate?

        let pieces = self.all_pieces_of_color(self.color_to_move);

        let mut valid_moves: Vec<Move> = Vec::new();

        for (piece, square) in pieces.iter() {
            for to_square in self.possible_moves(*square, false)? {
                println!(
                    "Evaluating move {} from {} to {}",
                    square_symbol(piece),
                    square,
                    to_square
                );

                let moved_board = self.clone().move_piece(*square, to_square)?;

                // Cannot move into check
                if !moved_board.is_in_check(self.color_to_move)? {
                    valid_moves.push(Move {
                        from: *square,
                        to: to_square,
                        score: moved_board.evaluate_position(self.color_to_move)?,
                    })
                }
            }
        }

        // Find best move
        Ok(valid_moves.iter().max_by_key(|m| m.score).map(|m| *m))
    }

    /**
     * Evaluate the position for the given color.
     */
    fn evaluate_position(&self, color: Color) -> Result<i32, BoardError> {
        if self.checkmate(color.opposite())? {
            println!("Found checkmate");
            Ok(i32::MAX)
        } else {
            Ok(random::<i8>() as i32)
        }
    }

    fn checkmate(&self, color: Color) -> Result<bool, BoardError> {
        if !self.is_in_check(color)? {
            return Ok(false);
        }

        println!("In check. Evaluating for checkmate");

        let pieces = self.all_pieces_of_color(color);

        for (piece, square) in pieces.iter() {
            for to_square in self.possible_moves(*square, false)? {
                println!(
                    "Evaluating move {} from {} to {}",
                    square_symbol(piece),
                    square,
                    to_square
                );

                let moved_board = self.clone().move_piece(square.clone(), to_square)?;

                // At least one way out!
                if !moved_board.is_in_check(color)? {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in RANKS.iter().rev() {
            write!(f, "{} ", rank)?;

            for file in FILES.iter() {
                let piece: Option<Piece> =
                    Piece::from(self.board[Square::new(*file, *rank).index()]);

                let symbol = match piece {
                    Some(piece) => square_symbol(&piece),
                    None => ' ',
                };
                write!(f, "{} ", symbol)?;
            }
            write!(f, "\n")?;
        }

        for c in vec![' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] {
            write!(f, "{} ", c)?;
        }

        fmt::Result::Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct Move {
    from: Square,
    to: Square,
    score: i32,
}

fn square_symbol(p: &Piece) -> char {
    let Piece(piece, color) = *p;

    let uncolored = match piece & PIECE_MASK {
        EMPTY => ' ',
        PAWN => 'p',
        KNIGHT => 'n',
        BISHOP => 'b',
        ROOK => 'r',
        QUEEN => 'q',
        KING => 'k',
        unknown => panic!("Unknown piece {}", unknown),
    };

    match color {
        BLACK => uncolored,
        WHITE => uncolored.to_ascii_uppercase(),
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
struct Square(File, Rank);

impl Square {
    fn new(file: File, rank: Rank) -> Square {
        Square(file, rank)
    }

    fn from_index(i: usize) -> Square {
        if i >= N_SQUARES {
            panic!("Square index {} is larger than max {}", i, N_SQUARES);
        }

        let rank = Rank::from_index(i);
        let file = File::from_index(i);

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

impl TryFrom<&str> for Square {
    type Error = BoardError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let re = Regex::new(r"^([A-H])([1-8])$")?;

        let capture = match re.captures_iter(s).next() {
            Some(capture) => capture,
            None => return Err(ParseError(format!("Could not parse {} as a square", s))),
        };
        Ok(Square::new(
            File::from_str(&capture[1]),
            Rank::from_str(&capture[2]),
        ))
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

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Square(file, rank) = self;
        write!(f, "{}{}", file, rank)
    }
}

fn main() {
    let b = Board::empty()
        .place_piece(Piece(PAWN, WHITE), Square::new(File::A, Rank::_2))
        .place_piece(Piece(KNIGHT, WHITE), Square::new(File::B, Rank::_1))
        .place_piece(Piece(BISHOP, WHITE), Square::new(File::C, Rank::_1))
        .place_piece(Piece(KING, WHITE), Square::new(File::E, Rank::_1))
        .place_piece(Piece(BISHOP, BLACK), Square::new(File::C, Rank::_8))
        .place_piece(Piece(KING, BLACK), Square::new(File::E, Rank::_8));

    println!("{}", b);
    println!(
        "{}",
        b.move_piece(
            Square::new(File::A, Rank::_2),
            Square::new(File::A, Rank::_4)
        )
        .unwrap()
    );

    let b2 = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square::new(File::A, Rank::_2))
        .place_piece(Piece(KING, WHITE), Square::new(File::E, Rank::_1))
        .place_piece(Piece(ROOK, BLACK), Square::new(File::C, Rank::_8))
        .place_piece(Piece(KING, BLACK), Square::new(File::E, Rank::_8));

    println!("{}", b2);
    println!("{:?}", b2.find_next_move().unwrap());
}

#[cfg(test)]
fn sorted(mut v: Vec<Square>) -> Vec<Square> {
    v.sort();
    v
}

#[cfg(test)]
fn square(s: &str) -> Square {
    Square::try_from(s).unwrap()
}

#[test]
fn test_king_free_movement() {
    let board = Board::empty()
        .place_piece(Piece(KING, BLACK), Square(File::A, Rank::_1))
        .place_piece(Piece(KING, WHITE), Square(File::C, Rank::_6));

    assert_eq!(
        sorted(
            board
                .possible_moves(Square(File::A, Rank::_1), false)
                .unwrap()
        ),
        sorted(vec![
            (File::A, Rank::_2).into(),
            (File::B, Rank::_2).into(),
            (File::B, Rank::_1).into(),
        ])
    );

    assert_eq!(
        sorted(
            board
                .possible_moves(Square(File::C, Rank::_6), false)
                .unwrap()
        ),
        sorted(vec![
            (File::C, Rank::_7).into(),
            (File::D, Rank::_7).into(),
            (File::D, Rank::_6).into(),
            (File::D, Rank::_5).into(),
            (File::C, Rank::_5).into(),
            (File::B, Rank::_5).into(),
            (File::B, Rank::_6).into(),
            (File::B, Rank::_7).into(),
        ])
    );

    assert_eq!(
        sorted(
            Board::empty()
                .place_piece(Piece(KING, BLACK), Square(File::H, Rank::_8))
                .possible_moves(Square(File::H, Rank::_8), false)
                .unwrap()
        ),
        sorted(vec![
            (File::H, Rank::_7).into(),
            (File::G, Rank::_7).into(),
            (File::G, Rank::_8).into(),
        ])
    );
}

#[test]
fn test_king_obstructed_movement() {
    let board = Board::empty()
        .place_piece(Piece(KING, WHITE), Square(File::F, Rank::_2))
        .place_piece(Piece(PAWN, WHITE), Square(File::G, Rank::_3))
        .place_piece(Piece(PAWN, WHITE), Square(File::F, Rank::_3))
        .place_piece(Piece(PAWN, WHITE), Square(File::E, Rank::_2))
        .place_piece(Piece(PAWN, WHITE), Square(File::F, Rank::_1));

    assert_eq!(
        sorted(
            board
                .possible_moves(Square(File::F, Rank::_2), false)
                .unwrap()
        ),
        sorted(vec![
            (File::G, Rank::_2).into(),
            (File::G, Rank::_1).into(),
            (File::E, Rank::_1).into(),
            (File::E, Rank::_3).into(),
        ])
    );
}

#[test]
fn test_king_cannot_move_into_check() {
    let board = Board::empty()
        .place_piece(Piece(KING, BLACK), Square(File::A, Rank::_1))
        .place_piece(Piece(ROOK, WHITE), Square(File::C, Rank::_2));

    assert_eq!(
        sorted(
            board
                .possible_moves(Square(File::A, Rank::_1), false)
                .unwrap()
        ),
        sorted(vec![(File::B, Rank::_1).into(),])
    );
}

#[test]
fn test_king_in_check() {
    assert!(Board::empty()
        .place_piece(Piece(KING, WHITE), Square(File::F, Rank::_2))
        .place_piece(Piece(ROOK, BLACK), Square(File::F, Rank::_5))
        .is_in_check(WHITE)
        .unwrap());

    assert!(Board::empty()
        .place_piece(Piece(KING, WHITE), Square(File::F, Rank::_2))
        .place_piece(Piece(ROOK, BLACK), Square(File::A, Rank::_2))
        .is_in_check(WHITE)
        .unwrap());

    assert!(!Board::empty()
        .place_piece(Piece(KING, WHITE), Square(File::F, Rank::_2))
        .place_piece(Piece(ROOK, BLACK), Square(File::E, Rank::_5))
        .is_in_check(WHITE)
        .unwrap());
}

#[test]
fn test_rook_free_movement() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square(File::A, Rank::_1))
        .place_piece(Piece(ROOK, WHITE), Square(File::C, Rank::_6));

    assert_eq!(
        sorted(
            board
                .possible_moves(Square(File::A, Rank::_1), false)
                .unwrap()
        ),
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
        sorted(
            board
                .possible_moves(Square(File::C, Rank::_6), false)
                .unwrap()
        ),
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
fn test_rook_boundary_conditions() {
    let board = Board::empty().place_piece(Piece(ROOK, WHITE), Square(File::A, Rank::_8));

    assert_eq!(
        sorted(
            board
                .possible_moves(Square(File::A, Rank::_8), false)
                .unwrap()
        ),
        sorted(
            RANKS
                .iter()
                .filter(|r| **r != Rank::_8)
                .map(|r| Square(File::A, *r))
                .chain(
                    FILES
                        .iter()
                        .filter(|f| **f != File::A)
                        .map(|f| Square(*f, Rank::_8))
                )
                .collect()
        )
    )
}

#[test]
fn test_rook_obstructed_movement() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square(File::E, Rank::_5))
        .place_piece(Piece(PAWN, WHITE), Square(File::E, Rank::_2))
        .place_piece(Piece(PAWN, WHITE), Square(File::E, Rank::_6))
        .place_piece(Piece(PAWN, WHITE), Square(File::A, Rank::_5))
        .place_piece(Piece(PAWN, WHITE), Square(File::G, Rank::_5));

    assert_eq!(
        sorted(
            board
                .possible_moves(Square(File::E, Rank::_5), false)
                .unwrap()
        ),
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
fn test_rook_capture() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square(File::E, Rank::_5))
        .place_piece(Piece(PAWN, BLACK), Square(File::E, Rank::_2))
        .place_piece(Piece(PAWN, BLACK), Square(File::E, Rank::_6))
        .place_piece(Piece(PAWN, BLACK), Square(File::A, Rank::_5))
        .place_piece(Piece(PAWN, BLACK), Square(File::G, Rank::_5));

    assert_eq!(
        sorted(
            board
                .possible_moves(Square(File::E, Rank::_5), false)
                .unwrap()
        ),
        sorted(vec![
            // rank moves (up-down)
            (File::E, Rank::_2).into(),
            (File::E, Rank::_3).into(),
            (File::E, Rank::_4).into(),
            (File::E, Rank::_6).into(),
            // file moves (side-to-side)
            (File::A, Rank::_5).into(),
            (File::B, Rank::_5).into(),
            (File::C, Rank::_5).into(),
            (File::D, Rank::_5).into(),
            (File::F, Rank::_5).into(),
            (File::G, Rank::_5).into(),
        ])
    );
}

#[test]
fn test_is_empty() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square(File::A, Rank::_1))
        .place_piece(Piece(PAWN, BLACK), Square(File::C, Rank::_6));

    assert!(!board.is_empty(Square(File::A, Rank::_1).index()));
    assert!(board.is_occupied(Square(File::A, Rank::_1).index()));
    assert!(board.is_occupied_by_color(Square(File::A, Rank::_1).index(), WHITE));
    assert!(!board.is_occupied_by_color(Square(File::A, Rank::_1).index(), BLACK));

    assert!(!board.is_empty(Square(File::C, Rank::_6).index()));
    assert!(board.is_occupied(Square(File::C, Rank::_6).index()));
    assert!(!board.is_occupied_by_color(Square(File::C, Rank::_6).index(), WHITE));
    assert!(board.is_occupied_by_color(Square(File::C, Rank::_6).index(), BLACK));

    assert!(board.is_empty(Square(File::D, Rank::_3).index()));
    assert!(!board.is_occupied(Square(File::D, Rank::_3).index()));
    assert!(!board.is_occupied_by_color(Square(File::D, Rank::_3).index(), WHITE));
    assert!(!board.is_occupied_by_color(Square(File::D, Rank::_3).index(), BLACK));
}

#[test]
fn test_can_capture() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square(File::A, Rank::_1))
        .place_piece(Piece(PAWN, BLACK), Square(File::C, Rank::_6));

    assert!(!board.can_capture(Square(File::B, Rank::_7).index(), WHITE));
    assert!(!board.can_capture(Square(File::A, Rank::_1).index(), WHITE));
    assert!(board.can_capture(Square(File::C, Rank::_6).index(), WHITE));

    assert!(!board.can_capture(Square(File::B, Rank::_7).index(), BLACK));
    assert!(board.can_capture(Square(File::A, Rank::_1).index(), BLACK));
    assert!(!board.can_capture(Square(File::C, Rank::_6).index(), BLACK));
}

#[test]
fn test_checkmate() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square(File::A, Rank::_1))
        .place_piece(Piece(ROOK, WHITE), Square(File::B, Rank::_2))
        .place_piece(Piece(KING, BLACK), Square(File::A, Rank::_6));

    assert!(board.checkmate(BLACK).unwrap());
}

#[test]
fn test_can_escape_checkmate() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square(File::A, Rank::_1))
        .place_piece(Piece(ROOK, WHITE), Square(File::B, Rank::_2))
        .place_piece(Piece(KING, BLACK), Square(File::A, Rank::_6))
        .place_piece(Piece(ROOK, BLACK), Square(File::H, Rank::_5));

    // Rook can intervene on A5
    assert!(!board.checkmate(BLACK).unwrap());
}

#[test]
fn test_checkmate_opponent_twin_rooks() {
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), Square(File::H, Rank::_8))
        .place_piece(Piece(ROOK, WHITE), Square(File::C, Rank::_1))
        .place_piece(Piece(ROOK, WHITE), Square(File::B, Rank::_2))
        .place_piece(Piece(KING, BLACK), Square(File::A, Rank::_6));

    println!("{}", board);

    let mv = board.find_next_move().unwrap().unwrap();

    assert_eq!(mv.from, Square(File::C, Rank::_1));
    assert_eq!(mv.to, Square(File::A, Rank::_1));
}

#[test]
fn test_checkmate_opponent_king_and_rook() {
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), Square(File::B, Rank::_6))
        .place_piece(Piece(ROOK, WHITE), Square(File::C, Rank::_1))
        .place_piece(Piece(KING, BLACK), Square(File::A, Rank::_8));

    println!("{}", board);

    let mv = board.find_next_move().unwrap().unwrap();

    assert_eq!(mv.from, Square(File::C, Rank::_1));
    assert_eq!(mv.to, Square(File::C, Rank::_8));
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
