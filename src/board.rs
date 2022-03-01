use core::cmp::Ordering;
use log::{debug, info};
use std::cmp;
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

pub enum PieceEnum {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

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
            can_castle: CastleRights::none(),
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

    pub fn make_move(&self, mv: Move) -> Result<Board> {
        let mut new = match mv {
            Move::Single { from, to } => self._move_piece(from, to),
            Move::CastleKingside => self.castle_kingside(self.color_to_move),
            Move::CastleQueenside => self.castle_queenside(self.color_to_move),
        }?;

        new.color_to_move = self.color_to_move.opposite();
        Ok(new)
    }

    pub fn castle_kingside(&self, color: Color) -> Result<Board> {
        let mut new = self.clone();

        // TODO: ensure not attacked, still allowed

        if color == Color::WHITE {
            new.board[G1.index()] = new.board[E1.index()];
            new.board[F1.index()] = new.board[H1.index()];
            new.board[E1.index()] = EMPTY;
            new.board[H1.index()] = EMPTY;

            new.can_castle.kingside_white = false;
            new.can_castle.queenside_white = false;
        } else {
            panic!("Not implemented")
        }

        Ok(new)
    }

    pub fn castle_queenside(&self, color: Color) -> Result<Board> {
        let mut new = self.clone();

        // TODO: ensure not attacked, still allowed

        if color == Color::WHITE {
            new.board[C1.index()] = new.board[E1.index()];
            new.board[D1.index()] = new.board[A1.index()];
            new.board[A1.index()] = EMPTY;
            new.board[E1.index()] = EMPTY;

            new.can_castle.kingside_white = false;
            new.can_castle.queenside_white = false;
        } else {
            panic!("Not implemented")
        }

        Ok(new)
    }

    fn _move_piece(&self, from: Square, to: Square) -> Result<Board> {
        let mut new = self.clone();

        let i = from.index();
        let j = to.index();

        if new.board[i] & PIECE_MASK == EMPTY {
            return Err(NoPieceOnFromSquare(from));
        }

        new.board[j] = new.board[i];
        new.board[i] = EMPTY;

        // TODO: verify that move is valid.
        // TODO: update castling rights

        Ok(new)
    }

    fn is_occupied(&self, square: usize) -> bool {
        !self.is_empty(square)
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
                    .iter()
                    .find_map(|mv| match mv {
                        Move::Single { from: _, to } if to == &Square::from_index(square) => {
                            Some(to)
                        }
                        _ => None,
                    })
                    .is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn possible_moves(&self, from: Square, ignore_king_jeopardy: bool) -> Result<Vec<Move>> {
        // TODO: check color, turn

        match Piece::from(self.board[from.index()]) {
            None => Err(NoPieceOnFromSquare(from)),
            Some(p) if p.piece() == PAWN => Ok(cross_product(from, self._pawn_moves(from, &p))),
            Some(p) if p.piece() == KNIGHT => Ok(cross_product(from, self._knight_moves(from, &p))),
            Some(p) if p.piece() == BISHOP => Ok(cross_product(from, self._bishop_moves(from, &p))),
            Some(p) if p.piece() == ROOK => Ok(cross_product(from, self._rook_moves(from, &p))),
            Some(p) if p.piece() == QUEEN => Ok(cross_product(from, self._queen_moves(from, &p))),
            Some(p) if p.piece() == KING => self
                ._king_moves(from, &p, ignore_king_jeopardy)
                .map(|moves| cross_product(from, moves)),
            _ => Err(NotImplemented),
        }
    }

    // Return all moves possible for the given color, including castling and promotion
    fn all_moves(&self, color: Color) -> Result<Vec<Move>> {
        let mut moves: Vec<Move> = self
            .all_pieces_of_color(color) // TODO: self.color_to_move?
            .iter()
            .flat_map(move |(_, square)| {
                self.possible_moves(*square, false)
                    .unwrap() // TODO fix this
                    .into_iter()
            })
            .collect();

        if color == Color::WHITE {
            // TODO: must not be attacked
            if self.can_castle.kingside_white
                && self.is_empty(F1.index())
                && self.is_empty(G1.index())
            {
                moves.push(Move::CastleKingside)
            }

            // TODO: must not be attacked
            if self.can_castle.queenside_white
                && self.is_empty(B1.index())
                && self.is_empty(C1.index())
                && self.is_empty(D1.index())
            {
                moves.push(Move::CastleQueenside)
            }
        }

        Ok(moves)
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

        debug!("Main line score={}, path={:?}", score, path);
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
            debug!("Position is checkmate for {}", self.color_to_move);

            return match self.color_to_move {
                WHITE => Ok((None, Score::MIN.plus(1), path.into())),
                BLACK => Ok((None, Score::MAX.minus(1), path.into())),
            };
        }

        let all_moves = self.all_moves(self.color_to_move)?;

        debug!(
            "{}: {}: All moves for position: {}",
            self.color_to_move,
            path,
            all_moves
                .iter()
                .map(|mv| mv.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        );

        let mut best_move: Option<Move> = None;
        let mut best_score = match self.color_to_move {
            WHITE => Score::MIN,
            BLACK => Score::MAX,
        };
        let mut best_path: Vec<Move> = path.into();

        for mv in all_moves.iter() {
            let moved_board = self.make_move(*mv)?;
            let moved_path = path.append(&mv);

            debug!(
                "{}: Evaluating move {}. Initial α={} β={}",
                self.color_to_move, moved_path, alpha, beta
            );

            // Cannot move into check. This helps verify that the position
            // is not checkmate: if the recursive call below returns no
            // moves then the position is mate
            if moved_board.is_in_check(self.color_to_move)? {
                debug!(
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
                        best_move = Some(*mv);
                        best_path = mainline.clone();
                    }
                }
                BLACK => {
                    beta = cmp::min(beta, score);
                    if score < best_score {
                        best_score = score;
                        best_move = Some(*mv);
                        best_path = mainline.clone();
                    }
                }
            }

            debug!(
                "{}: Evaluated move {} {} score={} α={} β={}",
                self.color_to_move,
                path,
                //                square_symbol(piece),
                mv,
                score,
                alpha,
                beta
            );

            if alpha >= beta {
                debug!("Found α={} >= β={}. Pruning rest of node.", alpha, beta);
                return Ok((Some(*mv), score, mainline));
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
            debug!("Found checkmate of {}", BLACK);
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

        debug!("{}: In check. Evaluating for checkmate", color);

        for mv in self.all_moves(color)?.iter() {
            let moved_board = self.make_move(*mv)?;

            // At least one way out!
            if !moved_board.is_in_check(color)? {
                return Ok(false);
            }
        }

        debug!("{}: Found checkmate", color);

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
pub enum Move {
    CastleKingside,
    CastleQueenside,
    Single { from: Square, to: Square },
}

impl Move {
    pub fn new(from: Square, to: Square) -> Self {
        Move::Single { from: from, to: to }
    }
}

impl From<(Square, Square)> for Move {
    fn from((from, to): (Square, Square)) -> Self {
        Move::Single { from, to }
    }
}

impl PartialOrd for Move {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Move {
    fn cmp(&self, other: &Self) -> Ordering {
        match (*self, *other) {
            (Move::CastleKingside, Move::CastleKingside) => Ordering::Equal,
            (Move::CastleKingside, _) => Ordering::Greater,
            (Move::CastleQueenside, Move::CastleKingside) => Ordering::Less,
            (Move::CastleQueenside, Move::CastleQueenside) => Ordering::Equal,
            (Move::CastleQueenside, _) => Ordering::Greater,
            (
                Move::Single {
                    from: from1,
                    to: to1,
                },
                Move::Single {
                    from: from2,
                    to: to2,
                },
            ) => (from1, to1).cmp(&(from2, to2)),

            (_, Move::CastleKingside) => Ordering::Less,
            (_, Move::CastleQueenside) => Ordering::Less,
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Move::Single { from, to } => write!(f, "{}{}", from, to),
            Move::CastleKingside => write!(f, "0-0"),
            Move::CastleQueenside => write!(f, "0-0-0"),
        }
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
fn sorted(mut v: Vec<Move>) -> Vec<Move> {
    v.sort();
    v
}

fn cross_product(from: Square, to: Vec<Square>) -> Vec<Move> {
    to.iter()
        .map(|square| Move::Single {
            from: from,
            to: *square,
        })
        .collect()
}

#[test]
fn test_king_free_movement() {
    init();
    let board = Board::empty()
        .place_piece(Piece(KING, BLACK), A1)
        .place_piece(Piece(KING, WHITE), C6);

    assert_eq!(
        sorted(board.possible_moves(A1, false).unwrap()),
        sorted(cross_product(A1, vec![A2, B2, B1]))
    );

    assert_eq!(
        sorted(board.possible_moves(C6, false).unwrap()),
        sorted(cross_product(C6, vec![C7, D7, D6, D5, C5, B5, B6, B7]))
    );

    assert_eq!(
        sorted(
            Board::empty()
                .place_piece(Piece(KING, BLACK), H8)
                .possible_moves(H8, false)
                .unwrap()
        ),
        sorted(cross_product(H8, vec![H7, G7, G8]))
    );
}

#[test]
fn test_king_obstructed_movement() {
    init();
    let board = Board::empty()
        .place_piece(Piece(KING, WHITE), F2)
        .place_piece(Piece(PAWN, WHITE), G3)
        .place_piece(Piece(PAWN, WHITE), F3)
        .place_piece(Piece(PAWN, WHITE), E2)
        .place_piece(Piece(PAWN, WHITE), F1);

    assert_eq!(
        sorted(board.possible_moves(F2, false).unwrap()),
        sorted(cross_product(F2, vec![G2, G1, E1, E3]))
    );
}

#[test]
fn test_king_cannot_move_into_check() {
    init();
    let board = Board::empty()
        .place_piece(Piece(KING, BLACK), A1)
        .place_piece(Piece(ROOK, WHITE), C2);

    assert_eq!(
        sorted(board.possible_moves(A1, false).unwrap()),
        sorted(vec![(A1, B1).into()])
    );
}

#[test]
fn test_king_in_check() {
    init();
    assert!(Board::empty()
        .place_piece(Piece(KING, WHITE), F2)
        .place_piece(Piece(ROOK, BLACK), F5)
        .is_in_check(WHITE)
        .unwrap());

    assert!(Board::empty()
        .place_piece(Piece(KING, WHITE), F2)
        .place_piece(Piece(ROOK, BLACK), A2)
        .is_in_check(WHITE)
        .unwrap());

    assert!(!Board::empty()
        .place_piece(Piece(KING, WHITE), F2)
        .place_piece(Piece(ROOK, BLACK), E5)
        .is_in_check(WHITE)
        .unwrap());
}

#[test]
fn test_king_can_take_queen_to_escape_check() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, BLACK), B8)
        .place_piece(Piece(QUEEN, WHITE), A7);

    assert!(board.is_in_check(BLACK).unwrap());

    assert!(board
        .possible_moves(B8, false)
        .unwrap()
        .iter()
        .find(|&&s| s == (B8, A7).into())
        .is_some());
}

#[test]
fn test_castle_kingside_white() {
    init();

    let board =
        crate::fen::parse("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3")
            .unwrap();

    assert!(board
        .all_moves(Color::WHITE)
        .unwrap()
        .iter()
        .map(|mv| *mv)
        .collect::<Vec<Move>>()
        .contains(&Move::CastleKingside));

    let castled_board = board.make_move(Move::CastleKingside).unwrap();

    assert_eq!(
        castled_board,
        crate::fen::parse("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 3") // TODO: fix halfmove clock
            .unwrap()
    )
}

#[test]
fn test_castle_kingside_white_not_allowed_if_king_has_moved() {
    init();

    let board =
        crate::fen::parse("r1bqk2r/ppp2ppp/2np1n2/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R w kq - 0 6")
            .unwrap();

    assert!(!board
        .all_moves(Color::WHITE)
        .unwrap()
        .iter()
        .map(|mv| *mv)
        .collect::<Vec<Move>>()
        .contains(&Move::CastleKingside));
}

#[test]
fn test_castle_queenside_white() {
    init();

    let board =
        crate::fen::parse("r3kbnr/ppp1pppp/2nq4/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5")
            .unwrap();

    assert!(board
        .all_moves(Color::WHITE)
        .unwrap()
        .iter()
        .map(|mv| *mv)
        .collect::<Vec<Move>>()
        .contains(&Move::CastleQueenside));

    let castled_board = board.make_move(Move::CastleQueenside).unwrap();

    assert_eq!(
        castled_board,
        crate::fen::parse("r3kbnr/ppp1pppp/2nq4/3p1b2/3P1B2/2N5/PPPQPPPP/2KR1BNR b kq - 6 5") // TODO: fix halfmove clock
            .unwrap()
    )
}

#[test]
fn test_castle_queenside_white_not_allowed_if_king_has_moved() {
    init();

    let board =
        crate::fen::parse("r3kb1r/ppp1pp1p/2nq1np1/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w kq - 0 7")
            .unwrap();

    assert!(!board
        .all_moves(Color::WHITE)
        .unwrap()
        .iter()
        .map(|mv| *mv)
        .collect::<Vec<Move>>()
        .contains(&Move::CastleQueenside));
}

#[test]
fn test_rook_free_movement() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(ROOK, WHITE), C6);

    assert_eq!(
        sorted(board.possible_moves(A1, false).unwrap()),
        sorted(cross_product(
            A1,
            vec![
                A2, A3, A4, A5, A6, A7, A8, // rank moves (up-down)
                B1, C1, D1, E1, F1, G1, H1 // file moves (side-to-side)
            ]
        ))
    );

    assert_eq!(
        sorted(board.possible_moves(C6, false).unwrap()),
        sorted(cross_product(
            C6,
            vec![
                C1, C2, C3, C4, C5, C7, C8, // rank moves (up-down)
                A6, B6, D6, E6, F6, G6, H6 // file moves (side-to-side)
            ]
        ))
    );
}

#[test]
fn test_rook_boundary_conditions() {
    init();
    let board = Board::empty().place_piece(Piece(ROOK, WHITE), A8);

    assert_eq!(
        sorted(board.possible_moves(A8, false).unwrap()),
        sorted(cross_product(
            A8,
            vec![A1, A2, A3, A4, A5, A6, A7, B8, C8, D8, E8, F8, G8, H8]
        ))
    )
}

#[test]
fn test_rook_obstructed_movement() {
    init();
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), E5)
        .place_piece(Piece(PAWN, WHITE), E2)
        .place_piece(Piece(PAWN, WHITE), E6)
        .place_piece(Piece(PAWN, WHITE), A5)
        .place_piece(Piece(PAWN, WHITE), G5);

    assert_eq!(
        sorted(board.possible_moves(E5, false).unwrap()),
        sorted(cross_product(
            E5,
            vec![
                E3, E4, // rank moves (up-down)
                B5, C5, D5, F5 // file moves (side-to-side)
            ]
        ))
    );
}

#[test]
fn test_rook_capture() {
    init();
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), E5)
        .place_piece(Piece(PAWN, BLACK), E2)
        .place_piece(Piece(PAWN, BLACK), E6)
        .place_piece(Piece(PAWN, BLACK), A5)
        .place_piece(Piece(PAWN, BLACK), G5);

    assert_eq!(
        sorted(board.possible_moves(E5, false).unwrap()),
        sorted(cross_product(
            E5,
            vec![
                E2, E3, E4, E6, // rank moves (up-down)
                A5, B5, C5, D5, F5, G5 // file moves (side-to-side)
            ]
        ))
    );
}

#[test]
fn test_is_empty() {
    init();
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(PAWN, BLACK), C6);

    assert!(!board.is_empty(A1.index()));
    assert!(board.is_occupied(A1.index()));
    assert!(board.is_occupied_by_color(A1.index(), WHITE));
    assert!(!board.is_occupied_by_color(A1.index(), BLACK));

    assert!(!board.is_empty(C6.index()));
    assert!(board.is_occupied(C6.index()));
    assert!(!board.is_occupied_by_color(C6.index(), WHITE));
    assert!(board.is_occupied_by_color(C6.index(), BLACK));

    assert!(board.is_empty(D3.index()));
    assert!(!board.is_occupied(D3.index()));
    assert!(!board.is_occupied_by_color(D3.index(), WHITE));
    assert!(!board.is_occupied_by_color(D3.index(), BLACK));
}

#[test]
fn test_can_capture() {
    init();
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(PAWN, BLACK), C6);

    assert!(!board.can_capture(B7.index(), WHITE));
    assert!(!board.can_capture(A1.index(), WHITE));
    assert!(board.can_capture(C6.index(), WHITE));

    assert!(!board.can_capture(B7.index(), BLACK));
    assert!(board.can_capture(A1.index(), BLACK));
    assert!(!board.can_capture(C6.index(), BLACK));
}

#[test]
fn test_checkmate() {
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(ROOK, WHITE), B2)
        .place_piece(Piece(KING, BLACK), A6);

    assert!(board.checkmate(BLACK).unwrap());
}

#[test]
fn test_can_escape_checkmate() {
    init();
    let board = Board::empty()
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(ROOK, WHITE), B2)
        .place_piece(Piece(KING, BLACK), A6)
        .place_piece(Piece(ROOK, BLACK), H5);

    // Rook can intervene on A5
    assert!(!board.checkmate(BLACK).unwrap());
}

#[test]
fn test_checkmate_opponent_twin_rooks() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), H8)
        .place_piece(Piece(ROOK, WHITE), C1)
        .place_piece(Piece(ROOK, WHITE), B2)
        .place_piece(Piece(KING, BLACK), A6);

    println!("{}", board);

    let (mv, _) = board.find_next_move(1).unwrap().unwrap();

    assert_eq!(mv, Move::new(C1, A1));
}

#[test]
fn test_checkmate_opponent_king_and_rook() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), B6)
        .place_piece(Piece(ROOK, WHITE), C1)
        .place_piece(Piece(KING, BLACK), A8);

    println!("{}", board);

    let (mv, _) = board.find_next_move(1).unwrap().unwrap();

    assert_eq!(mv, Move::new(C1, C8));
}

#[test]
fn test_checkmate_opponent_king_and_rook_2_moves() {
    init();
    let board1 = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(ROOK, WHITE), A7)
        .place_piece(Piece(ROOK, WHITE), B7)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(ROOK, BLACK), F8);

    let (mv, _) = board1.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv, Move::new(B7, H7));

    let board2 = board1.make_move(mv).unwrap();

    // Only move available
    let board3 = board2.make_move((H8, G8).into()).unwrap();

    let (mv3, _) = board3.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv3, Move::new(A7, G7));

    let board4 = board3.make_move(mv3).unwrap();
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
        .place_piece(Piece(KING, BLACK), A1)
        .place_piece(Piece(ROOK, BLACK), A7)
        .place_piece(Piece(ROOK, BLACK), B7)
        .place_piece(Piece(KING, WHITE), H8)
        .place_piece(Piece(ROOK, WHITE), F8);

    let (mv, _) = board1.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv, Move::new(B7, H7));

    let board2 = board1.make_move(mv).unwrap();

    // Only move available
    let board3 = board2.make_move((H8, G8).into()).unwrap();

    let (mv3, _) = board3.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv3, Move::new(A7, G7));

    let board4 = board3.make_move(mv3).unwrap();
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
        .place_piece(Piece(KING, BLACK), A1)
        .place_piece(Piece(ROOK, BLACK), A7)
        .place_piece(Piece(KING, WHITE), H8)
        .place_piece(Piece(ROOK, WHITE), C7);

    println!("{}", board);

    let (mv, _) = board.find_next_move(3).unwrap().unwrap();

    assert_eq!(mv, Move::new(A7, C7));
}

#[test]
fn test_puzzle_capture_rook() {
    // From https://lichess.org/training/KXQEn

    init();

    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, BLACK), F3)
        .place_piece(Piece(ROOK, BLACK), B3)
        .place_piece(Piece(KING, WHITE), D1)
        .place_piece(Piece(PAWN, WHITE), B2)
        .place_piece(Piece(PAWN, WHITE), F2)
        .place_piece(Piece(ROOK, WHITE), A3);

    Puzzle::new(board)
        .should_find_move(B3, A3)
        .should_find_move(B2, A3)
        .should_find_move(F3, F2);
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

        Self::new(self.board.make_move(found).unwrap())
    }

    fn respond_with(&self, from: Square, to: Square) -> Self {
        Self::new(self.board.make_move((from, to).into()).unwrap())
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
//         .should_find_move(F3, G5);

// }

#[test]
fn test_puzzle_grab_bishop_and_knight() {
    init();
    let board =
        crate::fen::parse("1kr4r/1p2qp2/p2p1p1p/4p3/4P3/1b2Q3/nPPRBPPP/1K5R w - - 0 19").unwrap();

    Puzzle::new(board)
        .should_find_move(E3, B3)
        .respond_with(A2, C3)
        .should_find_move(B2, C3);
}

#[test]
fn test_puzzle_grab_debug() {
    init();
    let board =
        crate::fen::parse("1kr4r/1p2qp2/p2p1p1p/4p3/4P3/1b2Q3/nPPRBPPP/1K5R w - - 0 19").unwrap();

    Puzzle::new(board.make_move((E3, A7).into()).unwrap()).should_find_move(B8, A7);
}

// TODO fix this test

#[test]
// fn test_puzzle_smothered_mate() {
//     // From https://www.chess.com/forum/view/endgames/endgame-puzzles2
//     // q1r4k/6pp/8/3Q2N1/8/8/6PP/7K w - - 0 1

//     init();

//     let board = Board::empty()
//         .with_color_to_move(WHITE)
//         .place_piece(Piece(QUEEN, BLACK), A8)
//         .place_piece(Piece(ROOK, BLACK), C8)
//         .place_piece(Piece(KING, BLACK), H8)
//         .place_piece(Piece(PAWN, BLACK), G7)
//         .place_piece(Piece(PAWN, BLACK), H7)
//         .place_piece(Piece(QUEEN, WHITE), D5)
//         .place_piece(Piece(KNIGHT, WHITE), G5)
//         .place_piece(Piece(PAWN, WHITE), G2)
//         .place_piece(Piece(PAWN, WHITE), H2)
//         .place_piece(Piece(KING, WHITE), H1);

//     Puzzle::new(board)
//         .should_find_move(G5, F7)
//         .respond_with(H8, G8)
//         .should_find_move(F7, H6)
//         .respond_with(G8, H8)
//         .should_find_move(D5, G8)
//         .respond_with(C8, G8)
//         .should_find_move(H6, F7)
//         .should_be_checkmate();
// }
#[test]
fn test_pawn_movement_from_start_rank_white() {
    init();
    let board = Board::empty().place_piece(Piece(PAWN, WHITE), A2);

    assert_eq!(
        sorted(board.possible_moves(A2, false).unwrap()),
        sorted(cross_product(A2, vec![A3, A4,]))
    );
}

#[test]
fn test_pawn_movement_from_start_rank_black() {
    init();
    let board = Board::empty().place_piece(Piece(PAWN, BLACK), C7);

    assert_eq!(
        sorted(board.possible_moves(C7, false).unwrap()),
        sorted(cross_product(C7, vec![C6, C5,]))
    );
}

#[test]
fn test_pawn_movement_blocked_from_start_rank_white() {
    init();
    let board1 = Board::empty()
        .place_piece(Piece(PAWN, WHITE), A2)
        .place_piece(Piece(ROOK, BLACK), A4);

    assert_eq!(
        sorted(board1.possible_moves(A2, false).unwrap()),
        sorted(vec![(A2, A3).into()])
    );

    let board2 = Board::empty()
        .place_piece(Piece(PAWN, WHITE), A2)
        .place_piece(Piece(ROOK, BLACK), A3);

    assert_eq!(
        sorted(board2.possible_moves(A2, false).unwrap()),
        Vec::new()
    );
}

#[test]
fn test_pawn_movement_blocked_from_start_rank_black() {
    init();
    let board1 = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), A7)
        .place_piece(Piece(ROOK, WHITE), A5);

    assert_eq!(
        sorted(board1.possible_moves(A7, false).unwrap()),
        sorted(vec![(A7, A6).into()])
    );

    let board2 = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), A7)
        .place_piece(Piece(ROOK, WHITE), A6);

    assert_eq!(
        sorted(board2.possible_moves(A7, false).unwrap()),
        Vec::new()
    );
}

#[test]
fn test_pawn_movement_from_middle_board_white() {
    init();
    let board = Board::empty().place_piece(Piece(PAWN, WHITE), H3);

    assert_eq!(
        sorted(board.possible_moves(H3, false).unwrap()),
        sorted(vec![(H3, H4).into()])
    );
}

#[test]
fn test_pawn_movement_from_middle_board_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), D4);

    assert_eq!(
        sorted(board.possible_moves(D4, false).unwrap()),
        sorted(vec![(D4, D3).into()])
    );
}

#[test]
fn test_pawn_capture_white() {
    init();
    let board = Board::empty()
        .place_piece(Piece(PAWN, WHITE), F6)
        .place_piece(Piece(ROOK, BLACK), E7)
        .place_piece(Piece(ROOK, BLACK), G7);

    assert_eq!(
        sorted(board.possible_moves(F6, false).unwrap()),
        sorted(cross_product(F6, vec![E7, F7, G7]))
    );
}

#[test]
fn test_pawn_capture_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), C4)
        .place_piece(Piece(ROOK, WHITE), B3)
        .place_piece(Piece(ROOK, WHITE), D3);

    assert_eq!(
        sorted(board.possible_moves(C4, false).unwrap()),
        sorted(cross_product(C4, vec![B3, C3, D3]))
    );
}

#[test]
fn test_pawn_cannot_capture_own_pieces_white() {
    init();
    let board = Board::empty()
        .place_piece(Piece(PAWN, WHITE), F6)
        .place_piece(Piece(ROOK, WHITE), E7)
        .place_piece(Piece(ROOK, WHITE), G7);

    assert_eq!(
        sorted(board.possible_moves(F6, false).unwrap()),
        sorted(vec![(F6, F7).into()])
    );
}

#[test]
fn test_pawn_cannot_capture_own_pieces_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), C4)
        .place_piece(Piece(ROOK, BLACK), B3)
        .place_piece(Piece(ROOK, BLACK), D3);

    assert_eq!(
        sorted(board.possible_moves(C4, false).unwrap()),
        sorted(vec![(C4, C3).into()])
    );
}

#[test]
fn test_pawn_cannot_capture_around_edge_of_board() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(PAWN, WHITE), A6)
        .place_piece(Piece(ROOK, BLACK), H8)
        .place_piece(Piece(PAWN, WHITE), H3)
        .place_piece(Piece(ROOK, BLACK), A5);

    assert_eq!(
        sorted(board.possible_moves(A6, false).unwrap()),
        sorted(vec![(A6, A7).into()])
    );

    assert_eq!(
        sorted(board.possible_moves(H3, false).unwrap()),
        sorted(vec![(H3, H4).into()])
    );
}

// #[test]
// fn test_pawn_capture_en_passant_white() {
//     init();
//     let board = Board::empty()
// 	.with_color_to_move(BLACK)
//         .place_piece(Piece(PAWN, WHITE), E5)
//         .place_piece(Piece(PAWN, BLACK), D7)
// 	.move_piece(D7, D5).unwrap(); // set up en passant target

//     assert_eq!(
//         sorted(board.possible_moves(E5, false).unwrap()),
//         sorted(vec![E6, D6])
//     );
// }

#[test]
fn test_knight_moves() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KNIGHT, BLACK), D4);

    assert_eq!(
        sorted(board.possible_moves(D4, false).unwrap()),
        sorted(cross_product(D4, vec![E6, F5, F3, E2, C2, B3, B5, C6,]))
    );
}

#[test]
fn test_bishop_moves() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(BISHOP, WHITE), C5);

    assert_eq!(
        sorted(board.possible_moves(C5, false).unwrap()),
        sorted(cross_product(
            C5,
            vec![
                A3, B4, D6, E7, F8, // right diagonal
                A7, B6, D4, E3, F2, G1, // left diagonal
            ]
        ))
    );
}

#[test]
fn test_queen_moves() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(QUEEN, WHITE), C5);

    assert_eq!(
        sorted(board.possible_moves(C5, false).unwrap()),
        sorted(cross_product(
            C5,
            vec![
                A3, B4, D6, E7, F8, // right diagonal
                A7, B6, D4, E3, F2, G1, // left diagonal
                A5, B5, D5, E5, F5, G5, H5, // horizontal
                C1, C2, C3, C4, C6, C7, C8, // vertical
            ]
        ))
    );
}

#[test]
fn test_vector_transpose() {
    let cases = vec![
        (A1, MoveVector(1, 1), Some(B2)),
        (A1, MoveVector(0, 8), None),
        (A1, MoveVector(8, 0), None),
        (A1, MoveVector(1, -1), None),
        (C2, MoveVector(-1, 1), Some(B3)),
        (H8, MoveVector(-1, 1), None),
        (H8, MoveVector(-1, -1), Some(G7)),
        (F7, MoveVector(-1, 2), None),
        (B3, MoveVector(-2, 1), None),
        (D3, MoveVector(-2, 1), Some(B4)),
    ];

    for (start, vector, end) in cases.iter() {
        info!(
            "{} + {:?} = {}",
            start,
            vector,
            end.map(|e| format!("{}", e)).unwrap_or("None".to_string())
        );
        assert_eq!(Board::plus_vector(start, vector), *end);
    }
}
