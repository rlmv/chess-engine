use core::cmp::Ordering;
use log::info;
use std::cmp;
use std::convert::TryFrom;
use std::fmt;

use crate::color::*;
use crate::constants::*;
use crate::error::BoardError;
use crate::error::BoardError::*;
use crate::file::*;
use crate::rank::*;
use crate::square::*;
use crate::vector::*;

pub type Result<T> = std::result::Result<T, BoardError>;

/*
 * TODO:
 *
 * - implement other pieces
 * - ingest lichess puzzles in a test suite
 */

#[derive(Debug, Clone)]
pub struct Piece(pub u8, pub Color);

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

    pub fn from(x: u8) -> Option<Self> {
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

const EMPTY: u8 = 0b00000000;
pub const PAWN: u8 = 0b00000001;
pub const KNIGHT: u8 = 0b00000010;
pub const BISHOP: u8 = 0b00000011;
pub const ROOK: u8 = 0b000000100;
pub const QUEEN: u8 = 0b00000101;
pub const KING: u8 = 0b00000110;

const PIECE_MASK: u8 = 0b00000111;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct CastleRights {
    pub kingside_black: bool,
    pub queenside_black: bool,
    pub kingside_white: bool,
    pub queenside_white: bool,
}

impl CastleRights {
    pub fn all() -> Self {
        CastleRights {
            kingside_black: true,
            queenside_black: true,
            kingside_white: true,
            queenside_white: true,
        }
    }

    pub fn none() -> Self {
        CastleRights {
            kingside_black: false,
            queenside_black: false,
            kingside_white: false,
            queenside_white: false,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Board {
    board: [u8; 64],
    color_to_move: Color,
    en_passant_target: Option<Square>,
    // # of moves since last capture or pawn advance. For enforcing the 50-move rule.
    halfmove_clock: u16,
    // Move #, incremented after Black plays
    fullmove_clock: u16,
    can_castle: CastleRights,
}

impl Board {
    pub fn empty() -> Self {
        Board {
            board: [EMPTY; 64],
            color_to_move: WHITE,
            en_passant_target: None,
            halfmove_clock: 0,
            fullmove_clock: 1,
            can_castle: CastleRights::all(),
        }
    }

    pub fn with_color_to_move(&self, color: Color) -> Self {
        let mut new = self.clone();
        new.color_to_move = color;
        new
    }

    pub fn with_fullmove_clock(&self, i: u16) -> Self {
        let mut new = self.clone();
        new.fullmove_clock = i;
        new
    }

    pub fn with_halfmove_clock(&self, i: u16) -> Self {
        let mut new = self.clone();
        new.halfmove_clock = i;
        new
    }

    pub fn with_en_passant_target(&self, target: Option<Square>) -> Self {
        let mut new = self.clone();
        new.en_passant_target = target;
        new
    }

    pub fn with_castle_rights(&self, rights: CastleRights) -> Self {
        let mut new = self.clone();
        new.can_castle = rights;
        new
    }

    pub fn place_piece(&self, piece: Piece, on: Square) -> Board {
        let mut new = self.clone();
        new.board[on.index()] = piece.encode();
        new
    }

    pub fn move_piece(&self, from: Square, to: Square) -> Result<Board> {
        let mut new = self.clone();

        let i = from.index();
        let j = to.index();

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

        match Piece::from(self.board[from.index()]) {
            None => Err(NoPieceOnFromSquare(from)),
            Some(p) if p.piece() == PAWN => Ok(self._pawn_moves(from, &p)),
            Some(p) if p.piece() == KNIGHT => Ok(self._knight_moves(from, &p)),
            Some(p) if p.piece() == BISHOP => Ok(self._bishop_moves(from, &p)),
            Some(p) if p.piece() == ROOK => Ok(self._rook_moves(from, &p)),
            Some(p) if p.piece() == QUEEN => Ok(self._queen_moves(from, &p)),
            Some(p) if p.piece() == KING => self._king_moves(from, &p, ignore_king_jeopardy),
            _ => Err(NotImplemented),
        }
    }

    //
    fn plus_vector(s: &Square, v: &MoveVector) -> Option<Square> {
        let signed_index = s.index() as i8;

        let MoveVector(x, y) = *v;

        let target = signed_index + x + (y * N_FILES as i8);

        if x < 0 && (s.file().index() as i8 + x) < 0 {
            // ignore: wrap around to left
            None
        } else if x > 0 && (s.file().index() as i8 + x) >= N_FILES as i8 {
            // ignore: wrap around to right
            None
        } else if target < 0 {
            // out bottom of board
            None
        } else if target >= N_SQUARES as i8 {
            // out top
            None
        } else {
            Some(Square::from_index(target as usize))
        }
    }

    fn plus_vector_scaled(s: &Square, v: &MoveVector, max_magnitude: u8) -> Vec<Square> {
        let mut targets: Vec<Square> = Vec::new();

        for m in 1..=max_magnitude {
            let scaled_v = v.times(m);

            match Board::plus_vector(s, &scaled_v) {
                Some(t) => targets.push(t),
                None => break,
            }
        }
        targets
    }

    fn _pawn_moves(&self, from: Square, piece: &Piece) -> Vec<Square> {
        let mut moves: Vec<Square> = Vec::new();

        assert!(*from.rank() != Rank::_1);
        assert!(*from.rank() != Rank::_8);

        let (start_rank, move_vector, capture_vectors) = match piece.color() {
            WHITE => (
                Rank::_2,
                MoveVector(0, 1),
                [MoveVector(1, 1), MoveVector(-1, 1)],
            ),
            BLACK => (
                Rank::_7,
                MoveVector(0, -1),
                [MoveVector(1, -1), MoveVector(-1, -1)],
            ),
        };

        // forward moves (including from start rank)

        let max_magnitude = if *from.rank() == start_rank { 2 } else { 1 };

        for target in Board::plus_vector_scaled(&from, &move_vector, max_magnitude).iter() {
            if self.is_occupied(target.index()) {
                break;
            } else {
                moves.push(*target)
            }
        }

        // standard capture

        for target in capture_vectors
            .iter()
            .filter_map(|v| Board::plus_vector(&from, v))
        {
            if self.can_capture(target.index(), piece.color()) {
                moves.push(target);
            }
        }

        // TODO: en passant capture

        // TODO: promotion

        moves
    }

    // TODO: this "ignore_king_jeopardy" flag feels hacky. Is there a way to
    // simplify and avoid this? More efficient way to find all attackers of a
    // square?
    //
    // Maybe don't need to exclude moves into check? Can instead remove those
    // later with the is_in_check option?
    fn _king_moves(
        &self,
        from: Square,
        king: &Piece,
        ignore_king_jeopardy: bool,
    ) -> Result<Vec<Square>> {
        let mut moves: Vec<Square> = Vec::new();
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

        for target in MOVE_VECTORS
            .iter()
            .filter_map(|v| Board::plus_vector(&from, v))
        {
            if self.is_occupied_by_color(target.index(), king.color()) {
                // cannot move into square of own piece
                continue;
            } else if !ignore_king_jeopardy
                && self.attacked_by_color(target.index(), king.color().opposite())?
            {
                // Cannot move *into* check.
                // BUT: we don't care if we are testing when the opponents king
                // moves into check. Don't recurse.
                continue;
            } else {
                moves.push(target);
            }
        }

        // TODO: castle

        Ok(moves)
    }

    fn _rook_moves(&self, from: Square, rook: &Piece) -> Vec<Square> {
        let mut moves: Vec<Square> = Vec::new();

        const MOVE_VECTORS: [MoveVector; 4] = [
            MoveVector(1, 0),
            MoveVector(0, -1),
            MoveVector(-1, 0),
            MoveVector(0, 1),
        ];

        const MAX_MAGNITUDE: u8 = 7;

        // Iterate allowed vectors, scaling by all possible magnitudes
        for v in MOVE_VECTORS.iter() {
            for target in Board::plus_vector_scaled(&from, v, MAX_MAGNITUDE) {
                if self.is_occupied_by_color(target.index(), rook.color()) {
                    break;
                } else if self.can_capture(target.index(), rook.color()) {
                    moves.push(target);
                    break;
                } else {
                    moves.push(target);
                }
            }
        }

        moves
    }

    fn _knight_moves(&self, from: Square, knight: &Piece) -> Vec<Square> {
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

        MOVE_VECTORS
            .iter()
            .filter_map(|v| Board::plus_vector(&from, v))
            .filter(|target| !self.is_occupied_by_color(target.index(), knight.color()))
            .collect()
    }

    fn _bishop_moves(&self, from: Square, bishop: &Piece) -> Vec<Square> {
        let mut moves: Vec<Square> = Vec::new();

        const MOVE_VECTORS: [MoveVector; 4] = [
            MoveVector(1, 1),
            MoveVector(1, -1),
            MoveVector(-1, -1),
            MoveVector(-1, 1),
        ];

        const MAX_MAGNITUDE: u8 = 7;

        // Iterate allowed vectors, scaling by all possible magnitudes
        for v in MOVE_VECTORS.iter() {
            for target in Board::plus_vector_scaled(&from, v, MAX_MAGNITUDE) {
                if self.is_occupied_by_color(target.index(), bishop.color()) {
                    break;
                } else if self.can_capture(target.index(), bishop.color()) {
                    moves.push(target);
                    break;
                } else {
                    moves.push(target);
                }
            }
        }

        moves
    }

    fn _queen_moves(&self, from: Square, queen: &Piece) -> Vec<Square> {
        let mut moves = self._rook_moves(from, queen);
        moves.extend(self._bishop_moves(from, queen));
        moves
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

    pub fn find_next_move(&self, depth: u8) -> Result<Option<(Move, Score)>> {
        let (mv, score, path) =
            self._find_next_move(depth, &TraversalPath::head(), Score::MIN, Score::MAX)?;

        println!("Main line score={}, path={:?}", score, path);
        Ok(mv.map(|m| (m, score)))
    }

    fn _find_next_move(
        &self,
        depth: u8,
        path: &TraversalPath,
        mut alpha: Score,
        mut beta: Score,
    ) -> Result<(Option<Move>, Score, Vec<Move>)> {
        // Find all pieces
        // Generate all valid moves for those pieces.
        // After each move, must not be in check - prune.
        // Make each move - evaluate the position.
        // Pick highest scoring move

        // Evaluate stalemate?

        if depth == 0 {
            return Ok((None, self.evaluate_position()?, path.into()));
        } else if self.checkmate(self.color_to_move)? {
            info!("Position is checkmate for {}", self.color_to_move);

            return match self.color_to_move {
                WHITE => Ok((None, Score::MIN.plus(1), path.into())),
                BLACK => Ok((None, Score::MAX.minus(1), path.into())),
            };
        }

        let pieces = self.all_pieces_of_color(self.color_to_move);

        let all_moves: Vec<(Piece, Square, Square)> = pieces
            .iter()
            .flat_map(move |(piece, square)| {
                self.possible_moves(*square, false)
                    .unwrap() // TODO fix this
                    .into_iter()
                    .map(move |to_square| (piece.clone(), square.clone(), to_square.clone()))
            })
            .collect();

        info!(
            "{}: {}: All moves for position: {:?}",
            self.color_to_move,
            path,
            all_moves
                .iter()
                .map(|(_, from, to)| Move {
                    from: *from,
                    to: *to
                })
                .collect::<Vec<Move>>()
        );

        let mut best_move: Option<Move> = None;
        let mut best_score = match self.color_to_move {
            WHITE => Score::MIN,
            BLACK => Score::MAX,
        };
        let mut best_path: Vec<Move> = path.into();

        for (piece, square, to_square) in all_moves.iter() {
            let mv = Move {
                from: *square,
                to: *to_square,
            };

            let moved_board = self.move_piece(*square, *to_square)?;
            let moved_path = path.append(&mv);

            info!(
                "{}: Evaluating move {}. Initial α={} β={}",
                self.color_to_move, moved_path, alpha, beta
            );

            // Cannot move into check. This helps verify that the position
            // is not checkmate: if the recursive call below returns no
            // moves then the position is mate
            if moved_board.is_in_check(self.color_to_move)? {
                info!(
                    "{}: Continue. In check after move {}{}",
                    self.color_to_move, moved_path, mv
                );
                continue;
            }

            // TODO: adjust the above for stalemate

            // Evaluate board score at leaf nodes
            let (_, score, mainline) =
                moved_board._find_next_move(depth - 1, &moved_path, alpha, beta)?;

            // Ensure that only *fully-searched* paths are returned. A pruned path could
            // possibly be worse than the fully searched path. The alpha-beta bound guarantees
            // that pruned subtrees are *equivalent to or worse than* the original search path.

            match self.color_to_move {
                WHITE => {
                    alpha = cmp::max(alpha, score);
                    if score > best_score {
                        best_score = score;
                        best_move = Some(mv);
                        best_path = mainline.clone();
                    }
                }
                BLACK => {
                    beta = cmp::min(beta, score);
                    if score < best_score {
                        best_score = score;
                        best_move = Some(mv);
                        best_path = mainline.clone();
                    }
                }
            }

            info!(
                "{}: Evaluated move {} {} {} score={} α={} β={}",
                self.color_to_move,
                path,
                square_symbol(piece),
                mv,
                score,
                alpha,
                beta
            );

            if alpha >= beta {
                info!("Found α={} >= β={}. Pruning rest of node.", alpha, beta);
                return Ok((Some(mv), score, mainline));
            }
            //            }
        }

        Ok((best_move, best_score, best_path))
    }

    /*
     * Evaluate the position for the given color.
     *
     * Positive values favor white, negative favor black.
     */
    fn evaluate_position(&self) -> Result<Score> {
        if self.checkmate(BLACK)? {
            info!("Found checkmate of {}", BLACK);
            return Ok(Score::MAX.minus(1));
        } else if self.checkmate(WHITE)? {
            info!("Found checkmate of {}", WHITE);
            return Ok(Score::MIN.plus(1));
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

        Ok(Score(white_value - black_value))
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

        info!("{}: Found checkmate", color);

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

// Linked list containing the current path to the root in the minimax tree
// traversal
#[derive(Debug, PartialEq, Eq)]
enum TraversalPath<'a> {
    Head,
    Node(&'a TraversalPath<'a>, &'a Move),
}

impl<'a> TraversalPath<'a> {
    fn head() -> Self {
        TraversalPath::Head
    }

    fn append(&'a self, mv: &'a Move) -> Self {
        TraversalPath::Node(self, mv)
    }

    fn fold_left<T>(&self, zero: T, f: fn(accum: T, mv: &Move) -> T) -> T {
        match self {
            TraversalPath::Head => zero,
            TraversalPath::Node(next, mv) => f(next.fold_left(zero, f), mv),
        }
    }
}

impl Into<Vec<Move>> for &TraversalPath<'_> {
    fn into(self) -> Vec<Move> {
        self.fold_left(Vec::new(), |mut v, mv| {
            v.push(*mv);
            v
        })
    }
}

impl fmt::Display for TraversalPath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.fold_left(String::new(), |mut accum, mv| {
            accum.extend(mv.to_string().chars());
            accum.extend(" ".chars());
            accum
        });

        write!(f, "{}", s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: Square,
    pub to: Square,
}

impl Move {
    pub fn new(from: Square, to: Square) -> Self {
        Move { from: from, to: to }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from, self.to)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Score(i32);

impl Score {
    pub const MAX: Score = Score(i32::MAX);
    pub const MIN: Score = Score(i32::MIN);

    fn minus(&self, x: i32) -> Score {
        let Score(y) = self;
        Score(y - x)
    }

    fn plus(&self, x: i32) -> Score {
        let Score(y) = self;
        Score(y + x)
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        let &Score(i) = self;
        let &Score(j) = other;
        i.cmp(&j)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Score(i) = self;
        write!(f, "{}", i)
    }
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
fn test_king_can_take_queen_to_escape_check() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, BLACK), square("B8"))
        .place_piece(Piece(QUEEN, WHITE), square("A7"));
    // .is_in_check(WHITE)
    // .unwrap());

    assert!(board.is_in_check(BLACK).unwrap());

    println!("{:?}", board.possible_moves(square("B8"), false).unwrap());

    assert!(board
        .possible_moves(square("B8"), false)
        .unwrap()
        .iter()
        .find(|&&s| s == square("A7"))
        .is_some());
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

    let (mv, _) = board.find_next_move(1).unwrap().unwrap();

    assert_eq!(
        mv,
        Move::new(Square(File::C, Rank::_1), Square(File::A, Rank::_1))
    );
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

    let (mv, _) = board.find_next_move(1).unwrap().unwrap();

    assert_eq!(
        mv,
        Move::new(Square(File::C, Rank::_1), Square(File::C, Rank::_8))
    );
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

    let (mv, _) = board1.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv, Move::new(square("B7"), square("H7")));

    let board2 = board1.move_piece(mv.from, mv.to).unwrap();

    // Only move available
    let board3 = board2.move_piece(square("H8"), square("G8")).unwrap();

    let (mv3, _) = board3.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv3, Move::new(square("A7"), square("G7")));

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

    let (mv, _) = board1.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv, Move::new(square("B7"), square("H7")));

    let board2 = board1.move_piece(mv.from, mv.to).unwrap();

    // Only move available
    let board3 = board2.move_piece(square("H8"), square("G8")).unwrap();

    let (mv3, _) = board3.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv3, Move::new(square("A7"), square("G7")));

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

    let (mv, _) = board.find_next_move(3).unwrap().unwrap();

    assert_eq!(mv, Move::new(square("A7"), square("C7")));
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

    Puzzle::new(board)
        .should_find_move(square("B3"), square("A3"))
        .should_find_move(square("B2"), square("A3"))
        .should_find_move(square("F3"), square("F2"));
}

#[cfg(test)]
struct Puzzle {
    board: Board,
}

#[cfg(test)]
impl Puzzle {
    fn new(board: Board) -> Self {
        println!("{}", board);
        Puzzle { board: board }
    }

    fn should_find_move(&self, expect_from: Square, expect_to: Square) -> Self {
        let (found, _) = self.board.find_next_move(4).unwrap().unwrap();
        assert_eq!(found, Move::new(expect_from, expect_to));

        Self::new(self.board.move_piece(found.from, found.to).unwrap())
    }

    fn respond_with(&self, from: Square, to: Square) -> Self {
        Self::new(self.board.move_piece(from, to).unwrap())
    }

    fn should_be_checkmate(&self) {
        assert!(self.board.checkmate(self.board.color_to_move).unwrap());
    }
}

// #[test]
// fn test_puzzle_capture_it_all() {
//     // https://lichess.org/training/GDfDV
//     // TODO: too deep.
//     init();
//     let board = fen::parse("r2q3r/pp2pkbp/2n1b1p1/2p5/6n1/2N2N2/PPPP2PP/R1BQ1RK1 w - - 0 12").unwrap();

//     Puzzle::new(board)
//         .should_find_move(square("F3"), square("G5"));

// }

#[test]
fn test_puzzle_grab_bishop_and_knight() {
    init();
    let board =
        crate::fen::parse("1kr4r/1p2qp2/p2p1p1p/4p3/4P3/1b2Q3/nPPRBPPP/1K5R w - - 0 19").unwrap();

    Puzzle::new(board)
        .should_find_move(square("E3"), square("B3"))
        .respond_with(square("A2"), square("C3"))
        .should_find_move(square("B2"), square("C3"));
}

#[test]
fn test_puzzle_grab_debug() {
    init();
    let board =
        crate::fen::parse("1kr4r/1p2qp2/p2p1p1p/4p3/4P3/1b2Q3/nPPRBPPP/1K5R w - - 0 19").unwrap();

    Puzzle::new(board.move_piece(square("E3"), square("A7")).unwrap())
        .should_find_move(square("B8"), square("A7"));
    // Puzzle::new(board)
    //     .should_find_move(square("E3"), square("B3"))
    //     .respond_with(square("A2"), square("C3"))
    //     .should_find_move(square("B2"), square("C3"));
}

#[test]
fn test_puzzle_smothered_mate() {
    // From https://www.chess.com/forum/view/endgames/endgame-puzzles2
    // q1r4k/6pp/8/3Q2N1/8/8/6PP/7K w - - 0 1

    init();

    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(QUEEN, BLACK), square("A8"))
        .place_piece(Piece(ROOK, BLACK), square("C8"))
        .place_piece(Piece(KING, BLACK), square("H8"))
        .place_piece(Piece(PAWN, BLACK), square("G7"))
        .place_piece(Piece(PAWN, BLACK), square("H7"))
        .place_piece(Piece(QUEEN, WHITE), square("D5"))
        .place_piece(Piece(KNIGHT, WHITE), square("G5"))
        .place_piece(Piece(PAWN, WHITE), square("G2"))
        .place_piece(Piece(PAWN, WHITE), square("H2"))
        .place_piece(Piece(KING, WHITE), square("H1"));

    Puzzle::new(board)
        .should_find_move(square("G5"), square("F7"))
        .respond_with(square("H8"), square("G8"))
        .should_find_move(square("F7"), square("H6"))
        .respond_with(square("G8"), square("H8"))
        .should_find_move(square("D5"), square("G8"))
        .respond_with(square("C8"), square("G8"))
        .should_find_move(square("H6"), square("F7"))
        .should_be_checkmate();
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
fn test_pawn_movement_from_start_rank_black() {
    init();
    let board = Board::empty().place_piece(Piece(PAWN, BLACK), square("C7"));

    assert_eq!(
        sorted(board.possible_moves(square("C7"), false).unwrap()),
        sorted(vec![square("C6"), square("C5"),])
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
fn test_pawn_movement_blocked_from_start_rank_black() {
    init();
    let board1 = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), square("A7"))
        .place_piece(Piece(ROOK, WHITE), square("A5"));

    assert_eq!(
        sorted(board1.possible_moves(square("A7"), false).unwrap()),
        sorted(vec![square("A6")])
    );

    let board2 = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), square("A7"))
        .place_piece(Piece(ROOK, WHITE), square("A6"));

    assert_eq!(
        sorted(board2.possible_moves(square("A7"), false).unwrap()),
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
fn test_pawn_movement_from_middle_board_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), square("D4"));

    assert_eq!(
        sorted(board.possible_moves(square("D4"), false).unwrap()),
        sorted(vec![square("D3")])
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
fn test_pawn_capture_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), square("C4"))
        .place_piece(Piece(ROOK, WHITE), square("B3"))
        .place_piece(Piece(ROOK, WHITE), square("D3"));

    assert_eq!(
        sorted(board.possible_moves(square("C4"), false).unwrap()),
        sorted(vec![square("B3"), square("C3"), square("D3")])
    );
}

#[test]
fn test_pawn_cannot_capture_own_pieces_white() {
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
fn test_pawn_cannot_capture_own_pieces_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), square("C4"))
        .place_piece(Piece(ROOK, BLACK), square("B3"))
        .place_piece(Piece(ROOK, BLACK), square("D3"));

    assert_eq!(
        sorted(board.possible_moves(square("C4"), false).unwrap()),
        sorted(vec![square("C3")])
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
fn test_knight_moves() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KNIGHT, BLACK), square("D4"));

    assert_eq!(
        sorted(board.possible_moves(square("D4"), false).unwrap()),
        sorted(vec![
            square("E6"),
            square("F5"),
            square("F3"),
            square("E2"),
            square("C2"),
            square("B3"),
            square("B5"),
            square("C6"),
        ])
    );
}

#[test]
fn test_bishop_moves() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(BISHOP, WHITE), square("C5"));

    assert_eq!(
        sorted(board.possible_moves(square("C5"), false).unwrap()),
        sorted(vec![
            // right diagonal
            square("A3"),
            square("B4"),
            square("D6"),
            square("E7"),
            square("F8"),
            // left diagonal
            square("A7"),
            square("B6"),
            square("D4"),
            square("E3"),
            square("F2"),
            square("G1"),
        ])
    );
}

#[test]
fn test_queen_moves() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(QUEEN, WHITE), square("C5"));

    assert_eq!(
        sorted(board.possible_moves(square("C5"), false).unwrap()),
        sorted(vec![
            // right diagonal
            square("A3"),
            square("B4"),
            square("D6"),
            square("E7"),
            square("F8"),
            // left diagonal
            square("A7"),
            square("B6"),
            square("D4"),
            square("E3"),
            square("F2"),
            square("G1"),
            // horizontal
            square("A5"),
            square("B5"),
            square("D5"),
            square("E5"),
            square("F5"),
            square("G5"),
            square("H5"),
            // vertical
            square("C1"),
            square("C2"),
            square("C3"),
            square("C4"),
            square("C6"),
            square("C7"),
            square("C8"),
        ])
    );
}

#[test]
fn test_vector_transpose() {
    let cases = vec![
        ("A1", MoveVector(1, 1), Some("B2")),
        ("A1", MoveVector(0, 8), None),
        ("A1", MoveVector(8, 0), None),
        ("A1", MoveVector(1, -1), None),
        ("C2", MoveVector(-1, 1), Some("B3")),
        ("H8", MoveVector(-1, 1), None),
        ("H8", MoveVector(-1, -1), Some("G7")),
        ("F7", MoveVector(-1, 2), None),
        ("B3", MoveVector(-2, 1), None),
        ("D3", MoveVector(-2, 1), Some("B4")),
    ];

    for (start, vector, end) in cases.iter() {
        println!("{} + {:?} = {}", start, vector, end.unwrap_or("None"));
        assert_eq!(
            Board::plus_vector(&square(start), vector),
            end.map(|e| square(e))
        );
    }
}
