use core::cmp::Ordering;
use log::{debug, error, info};
use regex::Regex;
use std::convert::TryFrom;
use std::fmt;

/*
 * TODO:
 *
 * - implement other pieces
 * - ingest lichess puzzles in a test suite
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

    // Relative values of each piece
    fn value(&self) -> i32 {
        let Piece(piece, _) = *self;
        match piece & PIECE_MASK {
            PAWN => 100,
            KNIGHT => 300,
            BISHOP => 300,
            ROOK => 500,
            QUEEN => 900,
            KING => 100000,
            _ => panic!("Unknown piece {}", piece),
        }
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

type Result<T> = std::result::Result<T, BoardError>;

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
    en_passant_target: Option<Square>,
}

impl Board {
    fn empty() -> Self {
        Board {
            board: [EMPTY; 64],
            color_to_move: WHITE,
            en_passant_target: None,
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

    fn move_piece(&self, from: Square, to: Square) -> Result<Board> {
        let mut new = self.clone();

        let i = Board::square_index(&from);
        let j = Board::square_index(&to);

        if new.board[i] & PIECE_MASK == EMPTY {
            return Err(NoPieceOnFromSquare(from));
        }

        new.board[j] = new.board[i];
        new.board[i] = EMPTY;
        new.color_to_move = self.color_to_move.opposite();

        // TODO: verify that move is valid.
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

    fn is_in_check(&self, color: Color) -> Result<bool> {
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
    fn attacked_by_color(&self, square: usize, color: Color) -> Result<bool> {
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

    fn possible_moves(&self, from: Square, ignore_king_jeopardy: bool) -> Result<Vec<Square>> {
        // TODO: check color, turn

        match Piece::from(self.board[Board::square_index(&from)]) {
            None => Err(NoPieceOnFromSquare(from)),
            Some(p) if p.piece() == PAWN => Ok(self._pawn_moves(from, p)),
            Some(p) if p.piece() == ROOK => Ok(self._rook_moves(from, p)),
            Some(p) if p.piece() == KING => self._king_moves(from, p, ignore_king_jeopardy),
            _ => Err(NotImplemented),
        }
    }

    fn _pawn_moves(&self, from: Square, piece: Piece) -> Vec<Square> {
        // Move from initial rank

        let mut moves: Vec<Square> = Vec::new();

        let signed_index = Board::square_index(&from) as i8;

        if piece.color() == WHITE {
            assert!(*from.rank() != Rank::_1);
            assert!(*from.rank() != Rank::_8);

            let move_vectors = if *from.rank() == Rank::_2 {
                vec![(0, 1), (0, 2)]
            } else {
                vec![(0, 1)]
            };

            for (x, y) in move_vectors.iter() {
                let target = signed_index + x + (y * N_FILES as i8);

                if self.is_occupied(target as usize) {
                    break;
                } else {
                    moves.push(Square::from_index(target as usize));
                }
            }

            // standard_capture
            let capture_vectors: Vec<MoveVector> = vec![(1, 1), (-1, 1)];

            // TODO: en passant capture

            // TODO: bound checking out sides of board
            for (x, y) in capture_vectors.iter() {
                let target = signed_index + x + (y * N_FILES as i8);

		// top/bottom boundary checking is already handled by rank 1/8 assertion above

                if *x == -1 && target % N_FILES as i8 == (N_FILES as i8 - 1) {
                    // ignore: wrap around to left
                } else if *x == 1 && target % N_FILES as i8 == 0 {
                    // ignore: wrap around to right
                } else if self.can_capture(target as usize, piece.color()) {
                    moves.push(Square::from_index(target as usize));
                }
            }
        } else if piece.color() == BLACK {
            assert!(*from.rank() != Rank::_8);

            if *from.rank() == Rank::_7 {
                moves.push(Square(*from.file(), Rank::_6));
                moves.push(Square(*from.file(), Rank::_5));
            } else if *from.rank() != Rank::_1 {
                moves.push(Square::from_index(from.index() - N_FILES));
            }
        }

        // en-passant capture

        // blocked

        // promotion

        moves
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
    ) -> Result<Vec<Square>> {
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

    fn find_next_move(&self, depth: u8) -> Result<Option<Move>> {
        // Find all pieces
        // Generate all valid moves for those pieces.
        // After each move, must not be in check - prune.
        // Make each move - evaluate the position.
        // Pick highest scoring move

        // Evaluate stalemate?

        assert!(depth > 0);

        let pieces = self.all_pieces_of_color(self.color_to_move);

        let mut valid_moves: Vec<Move> = Vec::new();

        for (piece, square) in pieces.iter() {
            for to_square in self.possible_moves(*square, false)? {
                info!(
                    "{}: Evaluating move {} from {} to {}",
                    self.color_to_move,
                    square_symbol(piece),
                    square,
                    to_square
                );

                let moved_board = self.move_piece(*square, to_square)?;

                // Cannot move into check. This helps verify that the position
                // is not checkmate: if the recursive call below returns no
                // moves then the position is mate
                if moved_board.is_in_check(self.color_to_move)? {
                    continue;
                }

                // TODO: adjust the above for stalemate

                // Evaluate board score at leaf nodes
                let score = if depth == 1 {
                    moved_board.evaluate_position()?
                } else {
                    match (moved_board.find_next_move(depth - 1)?, self.color_to_move) {
                        (Some(mv), _) => mv.score,
                        // No moves, checkmate
                        (None, WHITE) => i32::MAX,
                        (None, BLACK) => i32::MIN,
                    }
                };

                valid_moves.push(Move {
                    from: *square,
                    to: to_square,
                    score: score,
                })
            }
        }

        // Find best move
        match self.color_to_move {
            WHITE => Ok(valid_moves.iter().max_by_key(|m| m.score).map(|m| *m)),
            BLACK => Ok(valid_moves.iter().min_by_key(|m| m.score).map(|m| *m)),
        }
    }

    /*
     * Evaluate the position for the given color.
     *
     * Positive values favor white, negative favor black.
     */
    fn evaluate_position(&self) -> Result<i32> {
        if self.checkmate(BLACK)? {
            info!("Found checkmate of {}", BLACK);
            return Ok(i32::MAX);
        } else if self.checkmate(WHITE)? {
            info!("Found checkmate of {}", WHITE);
            return Ok(i32::MIN);
        }

        let white_value: i32 = self
            .all_pieces_of_color(WHITE)
            .iter()
            .map(|(p, _)| p.value())
            .sum();
        let black_value: i32 = self
            .all_pieces_of_color(BLACK)
            .iter()
            .map(|(p, _)| p.value())
            .sum();

        Ok(white_value - black_value)
    }

    fn checkmate(&self, color: Color) -> Result<bool> {
        if !self.is_in_check(color)? {
            return Ok(false);
        }

        info!("{}: In check. Evaluating for checkmate", color);

        let pieces = self.all_pieces_of_color(color);

        for (_, square) in pieces.iter() {
            for to_square in self.possible_moves(*square, false)? {
                let moved_board = self.move_piece(square.clone(), to_square)?;

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

    fn rank(&self) -> &Rank {
        let Square(_, rank) = self;
        rank
    }

    fn file(&self) -> &File {
        let Square(file, _) = self;
        file
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

    fn try_from(s: &str) -> Result<Self> {
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
    env_logger::init();

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
    println!("{:?}", b2.find_next_move(1).unwrap());
}

#[cfg(test)]
fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
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
fn test_parse_square() {
    init();
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

#[test]
fn test_king_free_movement() {
    init();
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
    init();
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
    init();
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
    init();
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
    init();
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
    init();
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
    init();
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
    init();
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
    init();
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
    init();
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
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), Square(File::H, Rank::_8))
        .place_piece(Piece(ROOK, WHITE), Square(File::C, Rank::_1))
        .place_piece(Piece(ROOK, WHITE), Square(File::B, Rank::_2))
        .place_piece(Piece(KING, BLACK), Square(File::A, Rank::_6));

    println!("{}", board);

    let mv = board.find_next_move(1).unwrap().unwrap();

    assert_eq!(mv.from, Square(File::C, Rank::_1));
    assert_eq!(mv.to, Square(File::A, Rank::_1));
}

#[test]
fn test_checkmate_opponent_king_and_rook() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), Square(File::B, Rank::_6))
        .place_piece(Piece(ROOK, WHITE), Square(File::C, Rank::_1))
        .place_piece(Piece(KING, BLACK), Square(File::A, Rank::_8));

    println!("{}", board);

    let mv = board.find_next_move(1).unwrap().unwrap();

    assert_eq!(mv.from, Square(File::C, Rank::_1));
    assert_eq!(mv.to, Square(File::C, Rank::_8));
}

#[test]
fn test_checkmate_opponent_king_and_rook_2_moves() {
    init();
    let board1 = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), square("A1"))
        .place_piece(Piece(ROOK, WHITE), square("A7"))
        .place_piece(Piece(ROOK, WHITE), square("B7"))
        .place_piece(Piece(KING, BLACK), square("H8"))
        .place_piece(Piece(ROOK, BLACK), square("F8"));

    let mv = board1.find_next_move(3).unwrap().unwrap();
    assert_eq!((mv.from, mv.to), (square("B7"), square("H7")));

    let board2 = board1.move_piece(mv.from, mv.to).unwrap();

    // Only move available
    let board3 = board2.move_piece(square("H8"), square("G8")).unwrap();

    let mv3 = board3.find_next_move(3).unwrap().unwrap();
    assert_eq!((mv3.from, mv3.to), (square("A7"), square("G7")));

    let board4 = board3.move_piece(mv3.from, mv3.to).unwrap();
    assert!(board4.checkmate(BLACK).unwrap());

    println!("{}", board1);
    println!("{}", board2);
    println!("{}", board3);
    println!("{}", board4);
}

#[test]
fn test_checkmate_opponent_king_and_rook_2_moves_black_to_move() {
    init();
    let board1 = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, BLACK), square("A1"))
        .place_piece(Piece(ROOK, BLACK), square("A7"))
        .place_piece(Piece(ROOK, BLACK), square("B7"))
        .place_piece(Piece(KING, WHITE), square("H8"))
        .place_piece(Piece(ROOK, WHITE), square("F8"));

    let mv = board1.find_next_move(3).unwrap().unwrap();
    assert_eq!((mv.from, mv.to), (square("B7"), square("H7")));

    let board2 = board1.move_piece(mv.from, mv.to).unwrap();

    // Only move available
    let board3 = board2.move_piece(square("H8"), square("G8")).unwrap();

    let mv3 = board3.find_next_move(3).unwrap().unwrap();
    assert_eq!((mv3.from, mv3.to), (square("A7"), square("G7")));

    let board4 = board3.move_piece(mv3.from, mv3.to).unwrap();
    assert!(board4.checkmate(WHITE).unwrap());

    println!("{}", board1);
    println!("{}", board2);
    println!("{}", board3);
    println!("{}", board4);
}

#[test]
fn test_capture_free_piece() {
    init();

    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, BLACK), square("A1"))
        .place_piece(Piece(ROOK, BLACK), square("A7"))
        .place_piece(Piece(KING, WHITE), square("H8"))
        .place_piece(Piece(ROOK, WHITE), square("C7"));

    println!("{}", board);

    let mv = board.find_next_move(3).unwrap().unwrap();

    assert_eq!(mv.from, square("A7"));
    assert_eq!(mv.to, square("C7"));
}

#[test]
fn test_puzzle_capture_rook() {
    // From https://lichess.org/training/KXQEn

    init();

    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, BLACK), square("F3"))
        .place_piece(Piece(ROOK, BLACK), square("B3"))
        .place_piece(Piece(KING, WHITE), square("D1"))
        .place_piece(Piece(PAWN, WHITE), square("B2"))
        .place_piece(Piece(PAWN, WHITE), square("F2"))
        .place_piece(Piece(ROOK, WHITE), square("A3"));

    println!("{}", board);

    let mv = board.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv.from, square("B3"));
    assert_eq!(mv.to, square("A3"));

    let board2 = board.move_piece(mv.from, mv.to).unwrap();
    let mv2 = board2.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv2.from, square("B2"));
    assert_eq!(mv2.to, square("A3"));

    println!("{}", board2);

    let board3 = board2.move_piece(mv2.from, mv2.to).unwrap();
    let mv3 = board3.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv3.from, square("F3"));
    assert_eq!(mv3.to, square("F2"));

    println!("{}", board3);
}

#[test]
fn test_pawn_movement_from_start_rank_white() {
    init();
    let board = Board::empty().place_piece(Piece(PAWN, WHITE), square("A2"));

    assert_eq!(
        sorted(board.possible_moves(square("A2"), false).unwrap()),
        sorted(vec![square("A3"), square("A4"),])
    );
}

#[test]
fn test_pawn_movement_blocked_from_start_rank_white() {
    init();
    let board1 = Board::empty()
        .place_piece(Piece(PAWN, WHITE), square("A2"))
        .place_piece(Piece(ROOK, BLACK), square("A4"));

    assert_eq!(
        sorted(board1.possible_moves(square("A2"), false).unwrap()),
        sorted(vec![square("A3")])
    );

    let board2 = Board::empty()
        .place_piece(Piece(PAWN, WHITE), square("A2"))
        .place_piece(Piece(ROOK, BLACK), square("A3"));

    assert_eq!(
        sorted(board2.possible_moves(square("A2"), false).unwrap()),
        Vec::new()
    );
}

#[test]
fn test_pawn_movement_from_middle_board_white() {
    init();
    let board = Board::empty().place_piece(Piece(PAWN, WHITE), square("H3"));

    assert_eq!(
        sorted(board.possible_moves(square("H3"), false).unwrap()),
        sorted(vec![square("H4")])
    );
}

#[test]
fn test_pawn_capture_white() {
    init();
    let board = Board::empty()
        .place_piece(Piece(PAWN, WHITE), square("F6"))
        .place_piece(Piece(ROOK, BLACK), square("E7"))
        .place_piece(Piece(ROOK, BLACK), square("G7"));

    assert_eq!(
        sorted(board.possible_moves(square("F6"), false).unwrap()),
        sorted(vec![square("E7"), square("F7"), square("G7")])
    );
}

#[test]
fn test_pawn_cannot_capture_own_pieces() {
    init();
    let board = Board::empty()
        .place_piece(Piece(PAWN, WHITE), square("F6"))
        .place_piece(Piece(ROOK, WHITE), square("E7"))
        .place_piece(Piece(ROOK, WHITE), square("G7"));

    assert_eq!(
        sorted(board.possible_moves(square("F6"), false).unwrap()),
        sorted(vec![square("F7")])
    );
}

#[test]
fn test_pawn_cannot_capture_around_edge_of_board() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(PAWN, WHITE), square("A6"))
        .place_piece(Piece(ROOK, BLACK), square("H8"))
        .place_piece(Piece(PAWN, WHITE), square("H3"))
        .place_piece(Piece(ROOK, BLACK), square("A5"));

    assert_eq!(
        sorted(board.possible_moves(square("A6"), false).unwrap()),
        sorted(vec![square("A7")])
    );

    assert_eq!(
        sorted(board.possible_moves(square("H3"), false).unwrap()),
        sorted(vec![square("H4")])
    );
}

// #[test]
// fn test_pawn_capture_en_passant_white() {
//     init();
//     let board = Board::empty()
// 	.with_color_to_move(BLACK)
//         .place_piece(Piece(PAWN, WHITE), square("E5"))
//         .place_piece(Piece(PAWN, BLACK), square("D7"))
// 	.move_piece(square("D7"), square("D5")).unwrap(); // set up en passant target

//     assert_eq!(
//         sorted(board.possible_moves(square("E5"), false).unwrap()),
//         sorted(vec![square("E6"), square("D6")])
//     );
// }

#[test]
fn test_pawn_movement_from_start_rank_black() {
    init();
    let board = Board::empty().place_piece(Piece(PAWN, BLACK), square("C7"));

    assert_eq!(
        sorted(board.possible_moves(square("C7"), false).unwrap()),
        sorted(vec![square("C6"), square("C5"),])
    );
}

#[test]
fn test_pawn_movement_from_middle_board_black() {
    init();
    let board = Board::empty().place_piece(Piece(PAWN, WHITE), square("C4"));

    assert_eq!(
        sorted(board.possible_moves(square("C4"), false).unwrap()),
        sorted(vec![square("C5")])
    );
}
