use core::cmp::Ordering;
use log::{debug, info};
use std::cmp;
use std::fmt;

use crate::bitboard::*;
use crate::color::*;
use crate::constants::*;
use crate::error::BoardError;
use crate::error::BoardError::*;
use crate::file::*;
use crate::mv::*;
use crate::rank::*;
use crate::square::*;
use crate::vector::*;
use cached::proc_macro::cached;
pub type Result<T> = std::result::Result<T, BoardError>;

/*
 * TODO:
 * - ingest lichess puzzles in a test suite
 * https://github.com/AndyGrant/Ethereal/blob/master/src/movegen.c
 */

#[cached]
fn cached_all_moves(board: Board, color: Color) -> Result<Vec<Move>> {
    board.all_moves(color)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Copy)]
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

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
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

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct Board {
    board: [u8; 64],
    pub color_to_move: Color,
    en_passant_target: Option<Square>,
    // # of moves since last capture or pawn advance. For enforcing the 50-move rule.
    halfmove_clock: u16,
    // Move #, incremented after Black plays
    fullmove_clock: u16,
    can_castle: CastleRights,

    presence_white: Bitboard,
    presence_black: Bitboard,
    pawn_presence_white: PawnPresenceBitboard,
    pawn_presence_black: PawnPresenceBitboard,
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
            presence_white: Bitboard::empty(),
            presence_black: Bitboard::empty(),
            pawn_presence_white: PawnPresenceBitboard::empty(WHITE),
            pawn_presence_black: PawnPresenceBitboard::empty(BLACK),
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

        new.update_bitboards().unwrap(); // TODO no unwrap
        new
    }

    pub fn make_move(&self, mv: Move) -> Result<Board> {
        let mut new = self.clone();

        match mv {
            Move::Single { from, to } => new.move_piece(from, to),
            Move::Promote { from, to, piece } => new.promote_pawn(from, to, piece),
            Move::CastleKingside => new.castle_kingside(new.color_to_move),
            Move::CastleQueenside => new.castle_queenside(new.color_to_move),
        }?;

        // Clear old en passant target. If old and new are not equal that means
        // that new board has a new target and should be left alone.
        if new.en_passant_target == self.en_passant_target {
            new.en_passant_target = None;
        }

        new.color_to_move = self.color_to_move.opposite();

        if self.color_to_move == Color::BLACK {
            new.fullmove_clock += 1;
        }

        if self.is_pawn_advance(mv)? || self.is_capture(mv)? {
            new.halfmove_clock = 0;
        } else {
            new.halfmove_clock += 1
        }

        new.update_bitboards()?;

        Ok(new)
    }

    fn update_bitboards(&mut self) -> Result<()> {
        let mut pawn_presence_white = PawnPresenceBitboard::empty(WHITE);
        let mut pawn_presence_black = PawnPresenceBitboard::empty(BLACK);

        let mut presence_white = Bitboard::empty();
        let mut presence_black = Bitboard::empty();

        for (piece, square) in self.all_pieces() {
            match piece.color() {
                WHITE => {
                    if piece.piece() == PAWN {
                        pawn_presence_white = pawn_presence_white.set(square)
                    }
                    presence_white = presence_white.set(square)
                }
                BLACK => {
                    if piece.piece() == PAWN {
                        pawn_presence_black = pawn_presence_black.set(square)
                    }
                    presence_black = presence_black.set(square)
                }
            }
        }

        self.pawn_presence_white = pawn_presence_white;
        self.pawn_presence_black = pawn_presence_black;

        self.presence_white = presence_white;
        self.presence_black = presence_black;

        Ok(())
    }

    fn is_pawn_advance(&self, mv: Move) -> Result<bool> {
        match mv {
            Move::Single { from, to: _ } => {
                if let Some(Piece(piece, _)) = self.piece_on_square(from) {
                    Ok(piece == PAWN)
                } else {
                    Err(NoPieceOnFromSquare(from))
                }
            }
            Move::Promote {
                from: _,
                to: _,
                piece: _,
            } => Ok(true),
            Move::CastleKingside => Ok(false),
            Move::CastleQueenside => Ok(false),
        }
    }

    fn is_capture(&self, mv: Move) -> Result<bool> {
        // TODO include en passant capture here
        match mv {
            Move::Single { from: _, to }
            | Move::Promote {
                from: _,
                to,
                piece: _,
            } => match self.piece_on_square(to) {
                Some(Piece(_, target_color)) if target_color == self.color_to_move.opposite() => {
                    Ok(true)
                }
                Some(_) => Err(IllegalMove(
                    "Trying to capture piece of same color".to_string(),
                )),
                None => Ok(false),
            },
            Move::CastleKingside => Ok(false),
            Move::CastleQueenside => Ok(false),
        }
    }

    fn castle_kingside(&mut self, color: Color) -> Result<()> {
        if !self.can_castle_kingside(color)? {
            return Err(IllegalCastle);
        }

        if color == Color::WHITE {
            self.board[G1.index()] = self.board[E1.index()];
            self.board[F1.index()] = self.board[H1.index()];
            self.board[E1.index()] = EMPTY;
            self.board[H1.index()] = EMPTY;

            self.can_castle.kingside_white = false;
            self.can_castle.queenside_white = false;
        } else {
            self.board[G8.index()] = self.board[E8.index()];
            self.board[F8.index()] = self.board[H8.index()];
            self.board[E8.index()] = EMPTY;
            self.board[H8.index()] = EMPTY;

            self.can_castle.kingside_black = false;
            self.can_castle.queenside_black = false;
        }

        Ok(())
    }

    fn castle_queenside(&mut self, color: Color) -> Result<()> {
        if !self.can_castle_queenside(color)? {
            return Err(IllegalCastle);
        }

        if color == Color::WHITE {
            self.board[C1.index()] = self.board[E1.index()];
            self.board[D1.index()] = self.board[A1.index()];
            self.board[A1.index()] = EMPTY;
            self.board[E1.index()] = EMPTY;

            self.can_castle.kingside_white = false;
            self.can_castle.queenside_white = false;
        } else {
            self.board[C8.index()] = self.board[E8.index()];
            self.board[D8.index()] = self.board[A8.index()];
            self.board[A8.index()] = EMPTY;
            self.board[E8.index()] = EMPTY;

            self.can_castle.kingside_black = false;
            self.can_castle.queenside_black = false;
        }

        Ok(())
    }

    fn can_castle_kingside(&self, color: Color) -> Result<bool> {
        let allowed = match color {
            Color::WHITE => {
                self.can_castle.kingside_white
                    && self.is_empty(F1.index())
                    && self.is_empty(G1.index())
                    && !self.attacked_by_color(E1, color.opposite())?
                    && !self.attacked_by_color(F1, color.opposite())?
                    && !self.attacked_by_color(G1, color.opposite())?
            }
            Color::BLACK => {
                self.can_castle.kingside_black
                    && self.is_empty(F8.index())
                    && self.is_empty(G8.index())
                    && !self.attacked_by_color(E8, color.opposite())?
                    && !self.attacked_by_color(F8, color.opposite())?
                    && !self.attacked_by_color(G8, color.opposite())?
            }
        };

        Ok(allowed)
    }

    fn can_castle_queenside(&self, color: Color) -> Result<bool> {
        let allowed = match color {
            Color::WHITE => {
                self.can_castle.queenside_white
                    && self.is_empty(B1.index())
                    && self.is_empty(C1.index())
                    && self.is_empty(D1.index())
                    && !self.attacked_by_color(C1, color.opposite())?
                    && !self.attacked_by_color(D1, color.opposite())?
                    && !self.attacked_by_color(E1, color.opposite())?
            }
            Color::BLACK => {
                self.can_castle.queenside_black
                    && self.is_empty(B8.index())
                    && self.is_empty(C8.index())
                    && self.is_empty(D8.index())
                    && !self.attacked_by_color(C8, color.opposite())?
                    && !self.attacked_by_color(D8, color.opposite())?
                    && !self.attacked_by_color(E8, color.opposite())?
            }
        };

        Ok(allowed)
    }

    fn move_piece(&mut self, from: Square, to: Square) -> Result<()> {
        let i = from.index();
        let j = to.index();

        if let Some(Piece(piece, color)) = self.piece_on_square(from) {
            if color != self.color_to_move {
                return Err(IllegalMove(format!(
                    "{}{} moves a {} piece, but it is {}'s turn",
                    from, to, color, self.color_to_move
                )));
            }

            // Set en passant target
            if piece == PAWN
                && from.file() == to.file()
                && (from.rank().index() as i8 - to.rank().index() as i8).abs() == 2
            {
                // Because from and to are on the same file the square in
                // between is computed by finding the difference between the
                // absolute indices in the board array
                self.en_passant_target = Some(Square::from_index((from.index() + to.index()) / 2))

            // Capture en passant
            } else if piece == PAWN
                && self.en_passant_target == Some(to)
                && self
                    .en_passant_target
                    .map(|target| self.is_empty(target.index()))
                    .unwrap_or(false)
            {
                // The captured piece is one rank different than the target square
                let captured_square = match self.color_to_move {
                    WHITE => to.index() - N_FILES,
                    BLACK => to.index() + N_FILES,
                };

                assert!(!self.is_empty(captured_square));
                self.board[captured_square] = EMPTY;

            // Update castling
            } else if piece == ROOK && color == WHITE && from == A1 {
                self.can_castle.queenside_white = false
            } else if piece == ROOK && color == BLACK && from == A8 {
                self.can_castle.queenside_black = false
            } else if piece == ROOK && color == WHITE && from == H1 {
                self.can_castle.kingside_white = false
            } else if piece == ROOK && color == BLACK && from == H8 {
                self.can_castle.kingside_black = false
            } else if piece == KING && color == WHITE && from == E1 {
                self.can_castle.kingside_white = false;
                self.can_castle.queenside_white = false
            } else if piece == KING && color == BLACK && from == E8 {
                self.can_castle.kingside_black = false;
                self.can_castle.queenside_black = false
            }

            self.board[j] = self.board[i];
            self.board[i] = EMPTY;
        } else {
            return Err(NoPieceOnFromSquare(from));
        }

        // TODO: verify that move is valid.

        Ok(())
    }

    fn promote_pawn(&mut self, from: Square, to: Square, piece: Piece) -> Result<()> {
        assert!(piece.color() == self.color_to_move);
        assert!(self.piece_on_square(from).map(|p| p.color()) == Some(self.color_to_move));

        // TODO: check that this is actually a promoting move. Can we impose type constraints?

        let i = from.index();
        let j = to.index();

        if self.board[i] & PIECE_MASK == EMPTY {
            return Err(NoPieceOnFromSquare(from));
        }

        self.board[j] = piece.encode();
        self.board[i] = EMPTY;

        Ok(())
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

    pub fn piece_on_square(&self, square: Square) -> Option<Piece> {
        Piece::from(self.board[square.index()])
    }

    fn is_in_check(&self, color: Color) -> Result<bool> {
        let mut kings = self
            .all_pieces_of_color(color)
            .filter(|(p, _)| p.piece() == KING);

        let (king, king_square) = kings.next().ok_or(IllegalState(format!(
            "Board is missing KING of color {}",
            color
        )))?;

        if kings.next().is_some() {
            return Err(IllegalState(format!(
                "Board has more than on KING of color {}",
                color
            )));
        }

        self.attacked_by_color(king_square, king.color().opposite())
    }

    // attacking moves is a subset of other moves -

    fn attacked_by_color(&self, target_square: Square, color: Color) -> Result<bool> {
        let attacks_target = |square: &Square| *square == target_square;

        let target_bitboard = Bitboard::empty().set(target_square);

        for (p, from) in self.all_pieces_of_color(color) {
            if p.piece() == PAWN {
                let pawn = PawnPresenceBitboard::empty(p.color()).set(from);

                if !(pawn.attacks() & target_bitboard).is_empty() {
                    return Ok(true);
                }
            } else if p.piece() == KNIGHT
                && !(self._knight_attacks(from, &p) & target_bitboard).is_empty()
            {
                return Ok(true);
            } else if p.piece() == BISHOP && self._bishop_moves(from, &p).iter().any(attacks_target)
            {
                return Ok(true);
            } else if p.piece() == ROOK && self._rook_moves(from, &p).iter().any(attacks_target) {
                return Ok(true);
            } else if p.piece() == QUEEN && self._queen_moves(from, &p).iter().any(attacks_target) {
                return Ok(true);
            } else if p.piece() == KING
                && !(self._king_attacks(from, &p) & target_bitboard).is_empty()
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    // Return all candidate moves for single pieces, allowing illegal moves
    // (e.g., that move the king into check)
    fn candidate_moves(&self, from: Square) -> Result<Vec<Move>> {
        // TODO: check color, turn

        let target_moves = match Piece::from(self.board[from.index()]) {
            None => Err(NoPieceOnFromSquare(from))?,
            Some(p) if p.piece() == PAWN => self._pawn_moves(from, &p),
            Some(p) if p.piece() == KNIGHT => {
                cross_product(from, self._knight_attacks(from, &p).squares().collect())
            }
            Some(p) if p.piece() == BISHOP => cross_product(from, self._bishop_moves(from, &p)),
            Some(p) if p.piece() == ROOK => cross_product(from, self._rook_moves(from, &p)),
            Some(p) if p.piece() == QUEEN => cross_product(from, self._queen_moves(from, &p)),
            Some(p) if p.piece() == KING => {
                cross_product(from, self._king_attacks(from, &p).squares().collect())
            }
            _ => Err(NotImplemented)?,
        };

        Ok(target_moves)
    }

    // Return all legal moves for the given square, filtering out those that result in
    // illegal states
    fn legal_moves(&self, from: Square) -> Result<Vec<Move>> {
        // TODO assert that piece on square is color to move

        let mut moves = Vec::new();

        for mv in self.candidate_moves(from)?.into_iter() {
            let moved_board = self.make_move(mv)?;
            // Cannot move into check
            if !moved_board.is_in_check(self.color_to_move)? {
                moves.push(mv)
            }
        }

        Ok(moves)
    }

    // Return all legal moves possible for the given color, including castling and promotion
    fn all_moves(&self, color: Color) -> Result<Vec<Move>> {
        // if let Some(cached_moves) = move_cache.with(|mut c| c.get(self).clone()) {
        //     return Ok(*cached_moves);
        // }

        let mut moves: Vec<Move> = self
            .all_pieces_of_color(color) // TODO: self.color_to_move?
            .flat_map(move |(_, square)| {
                self.legal_moves(square)
                    .unwrap() // TODO fix this
                    .into_iter()
            })
            .collect();

        if self.can_castle_kingside(color)? {
            moves.push(Move::CastleKingside);
        };

        if self.can_castle_queenside(color)? {
            moves.push(Move::CastleQueenside)
        };

        //        move_cache.with(|c| c.put(*self, moves.clone()));

        moves.sort_by_cached_key(|mv| match mv {
            Move::Promote {
                from: _,
                to: _,
                piece: _,
            } => 1,
            Move::Single { from: _, to: _ } if self.is_capture(*mv).unwrap() => 2,
            Move::CastleKingside => 3,
            Move::CastleQueenside => 3,
            Move::Single { from: _, to: _ } => 4,
        });
        Ok(moves)
    }

    pub fn plus_vector(s: &Square, v: &MoveVector) -> Option<Square> {
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

    fn plus_vector_scaled<'a>(
        s: &'a Square,
        v: &'a MoveVector,
        max_magnitude: u8,
    ) -> impl Iterator<Item = Square> + 'a {
        (1..=max_magnitude)
            .map(|m| {
                let scaled_v = v.times(m);
                Board::plus_vector(s, &scaled_v)
            })
            .take_while(|s| s.is_some())
            .filter_map(|s| s)
    }

    fn _pawn_moves(&self, from: Square, pawn: &Piece) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::with_capacity(8);

        assert!(*from.rank() != Rank::_1);
        assert!(*from.rank() != Rank::_8);

        let start_rank = match pawn.color() {
            WHITE => Rank::_2,
            BLACK => Rank::_7,
        };

        fn promoting_moves(
            from: Square,
            target: Square,
            color: Color,
        ) -> impl Iterator<Item = Move> {
            [QUEEN, ROOK, BISHOP, KNIGHT]
                .into_iter()
                .map(move |new_piece| Move::Promote {
                    from: from,
                    to: target,
                    piece: Piece(new_piece, color),
                })
        }

        fn is_promoting_square(target: &Square) -> bool {
            *target.rank() == Rank::_1 || *target.rank() == Rank::_8
        }

        // TODO: is it possible to vectorize over all pawns using the full
        // presence bitboards?
        let this_piece = PawnPresenceBitboard::empty(pawn.color()).set(from);

        // forward moves (including from start rank)

        let max_magnitude = if *from.rank() == start_rank { 2 } else { 1 };

        for magnitude in 1..=max_magnitude {
            let moved_forward = match pawn.color() {
                WHITE => this_piece.b.north_by(magnitude),
                BLACK => this_piece.b.south_by(magnitude),
            };

            let not_blocked = moved_forward & !(self.presence_black | self.presence_white);

            // NOTE: only works because dealing with one pawn:
            match not_blocked.squares().next() {
                Some(target) if is_promoting_square(&target) => {
                    moves.extend(promoting_moves(from, target, pawn.color()))
                }
                Some(target) => moves.push(Move::Single {
                    from: from,
                    to: target,
                }),
                None => break, // blocked, don't try the next square
            }
        }

        // capture

        let en_passant_target = self
            .en_passant_target
            .map(|target| Bitboard::empty().set(target))
            .unwrap_or(Bitboard::empty());

        let captures = match pawn.color() {
            WHITE => this_piece.attacks() & (self.presence_black | en_passant_target),
            BLACK => this_piece.attacks() & (self.presence_white | en_passant_target),
        };

        for target in captures.squares() {
            // capture and promote
            if is_promoting_square(&target) {
                moves.extend(promoting_moves(from, target, pawn.color()));

            // standard capture or en passant
            } else {
                moves.push(Move::Single {
                    from: from,
                    to: target,
                });
            }
        }

        moves
    }

    // Note: does not exclude moves that put the king in check
    fn _king_attacks(&self, from: Square, king: &Piece) -> Bitboard {
        let attacks = PRECOMPUTED_BITBOARDS.king_moves[from.index()];

        let same_color = match king.color() {
            WHITE => self.presence_white,
            BLACK => self.presence_black,
        };

        attacks & !same_color
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

    fn _knight_attacks<'a>(&'a self, from: Square, knight: &'a Piece) -> Bitboard {
        let attacks = PRECOMPUTED_BITBOARDS.knight_moves[from.index()];

        let same_color = match knight.color() {
            WHITE => self.presence_white,
            BLACK => self.presence_black,
        };

        attacks & !same_color
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

    fn all_pieces<'a>(&'a self) -> impl Iterator<Item = (Piece, Square)> + 'a {
        self.board
            .iter()
            .enumerate()
            .filter_map(move |(i, p)| Piece::from(*p).map(|piece| (piece, Square::from_index(i))))
    }

    fn all_pieces_of_color<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Piece, Square)> + 'a {
        self.all_pieces()
            .filter(move |(piece, _)| piece.color() == color)
    }

    pub fn find_next_move(&self, depth: u8) -> Result<Option<(Move, Score)>> {
        let (mv, score, path, node_count) =
            self._find_next_move(depth, &TraversalPath::head(), Score::MIN, Score::MAX)?;

        info!(
            "Main line score={}, path={:?}, node_count={}",
            score, path, node_count
        );
        Ok(mv.map(|m| (m, score)))
    }

    fn _find_next_move(
        &self,
        depth: u8,
        path: &TraversalPath,
        alpha_arg: Score,
        beta_arg: Score,
    ) -> Result<(Option<Move>, Score, Vec<Move>, u64)> {
        // Find all pieces
        // Generate all valid moves for those pieces.
        // After each move, must not be in check - prune.
        // Make each move - evaluate the position.
        // Pick highest scoring move

        let mut alpha = alpha_arg;
        let mut beta = beta_arg;

        // Leaf node, we are done
        if depth == 0 {
            return Ok((None, self.evaluate_position()?, path.into(), 1));
        }

        let all_moves = cached_all_moves(*self, self.color_to_move)?;

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

        // No legal moves, this position is either checkmate or stalemate
        if all_moves.len() == 0 {
            if self.is_in_check(self.color_to_move)? {
                debug!("Position is checkmate for {}", self.color_to_move);
                return match self.color_to_move {
                    WHITE => Ok((None, Score::checkmate_white(), path.into(), 1)),
                    BLACK => Ok((None, Score::checkmate_black(), path.into(), 1)),
                };
            } else {
                debug!("Position is stalemate");
                return Ok((None, Score::ZERO, path.into(), 1));
            }
        }

        let mut best_move: Option<Move> = None;
        let mut best_score = match self.color_to_move {
            WHITE => Score::MIN,
            BLACK => Score::MAX,
        };
        let mut best_path: Vec<Move> = path.into();

        let mut node_count: u64 = 1; // number of nodes visited by all

        for mv in all_moves.iter() {
            let moved_board = self.make_move(*mv)?;
            let moved_path = path.append(&mv);

            debug!(
                "{}: Evaluating move {} initial score={} α={} β={}",
                self.color_to_move, moved_path, best_score, alpha, beta
            );

            // Cannot move into check. This helps verify that the position
            // is not checkmate: if the recursive call below returns no
            // moves then the position is mate
            //
            // TODO: this is confusing. Cleaner way to handle this? Filter all moves to only return legal moves?
            if moved_board.is_in_check(self.color_to_move)? {
                debug!(
                    "{}: Continue. In check after move {}{}",
                    self.color_to_move, moved_path, mv
                );
                continue;
            }

            // Evaluate board score at leaf nodes
            let (_, score, mainline, subtree_node_count) =
                moved_board._find_next_move(depth - 1, &moved_path, alpha, beta)?;

            node_count += subtree_node_count;

            // Ensure that only *fully-searched* paths are returned. A pruned path could
            // possibly be worse than the fully searched path. The alpha-beta bound guarantees
            // that pruned subtrees are *equivalent to or worse than* the original search path.

            match self.color_to_move {
                WHITE => {
                    alpha = cmp::max(alpha, score);
                    if score > best_score
                        // shortest line is best, this picks the quickest forced mate sequence
                        || (score == best_score && mainline.len() < best_path.len())
                    {
                        best_score = score;
                        best_move = Some(*mv);
                        best_path = mainline.clone();
                    }
                }
                BLACK => {
                    beta = cmp::min(beta, score);
                    if score < best_score
                        || (score == best_score && mainline.len() < best_path.len())
                    {
                        best_score = score;
                        best_move = Some(*mv);
                        best_path = mainline.clone();
                    }
                }
            }

            debug!(
                "{}: Evaluated move {} score={} α={} β={} nodes={}",
                self.color_to_move, moved_path, score, alpha, beta, subtree_node_count
            );

            if alpha >= beta {
                debug!("Found α={} >= β={}. Pruning rest of node.", alpha, beta);
                return Ok((Some(*mv), score, mainline, node_count));
            }
        }

        Ok((best_move, best_score, best_path, node_count))
    }

    /*
     * Evaluate the position for the given color.
     *
     * Positive values favor white, negative favor black.
     */
    fn evaluate_position(&self) -> Result<Score> {
        if self.checkmate(BLACK)? {
            debug!("Found checkmate of {}", BLACK);
            return Ok(Score::checkmate_black());
        } else if self.checkmate(WHITE)? {
            debug!("Found checkmate of {}", WHITE);
            return Ok(Score::checkmate_white());
        } else if self.stalemate(self.color_to_move)? {
            debug!("Found stalemate");
            return Ok(Score::ZERO);
        }

        let white_value: i32 = self
            .all_pieces_of_color(WHITE)
            .map(|(p, _)| p.value())
            .sum();

        let black_value: i32 = self
            .all_pieces_of_color(BLACK)
            .map(|(p, _)| p.value())
            .sum();

        let mut white_bonus: i32 = 0;
        let mut black_bonus: i32 = 0;

        const OFF_INITIAL_SQUARE: i32 = 50;

        const WHITE_INITIAL_SQUARES: [(Square, u8); 4] =
            [(B1, KNIGHT), (C1, BISHOP), (F1, BISHOP), (G1, KNIGHT)];

        for (square, piece) in WHITE_INITIAL_SQUARES.into_iter() {
            if self.piece_on_square(square) != Some(Piece(piece, WHITE)) {
                white_bonus += OFF_INITIAL_SQUARE;
            }
        }

        const BLACK_INITIAL_SQUARES: [(Square, u8); 4] =
            [(B8, KNIGHT), (C8, BISHOP), (F8, BISHOP), (G8, KNIGHT)];

        for (square, piece) in BLACK_INITIAL_SQUARES.into_iter() {
            if self.piece_on_square(square) != Some(Piece(piece, BLACK)) {
                black_bonus += OFF_INITIAL_SQUARE;
            }
        }

        Ok(Score(white_value - black_value + white_bonus - black_bonus))
    }

    fn checkmate(&self, color: Color) -> Result<bool> {
        if !self.is_in_check(color)? {
            return Ok(false);
        }

        debug!("{}: In check. Evaluating for checkmate", color);

        for mv in cached_all_moves(*self, color)?.iter() {
            let moved_board = self.make_move(*mv)?;

            // At least one way out!
            if !moved_board.is_in_check(color)? {
                return Ok(false);
            }
        }

        debug!("{}: Found checkmate", color);

        Ok(true)
    }

    fn stalemate(&self, color: Color) -> Result<bool> {
        if self.is_in_check(color)? {
            return Ok(false);
        }

        debug!("{}: Evaluating for stalemate", color);

        for mv in cached_all_moves(*self, color)?.iter() {
            let moved_board = self.make_move(*mv)?;

            // At least one move that avoids check
            if !moved_board.is_in_check(color)? {
                return Ok(false);
            }
        }

        debug!("{}: No moves, stalemate", color);

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

pub fn square_symbol(p: &Piece) -> char {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Score(i32);

impl Score {
    pub const MAX: Score = Score(i32::MAX);
    pub const MIN: Score = Score(i32::MIN);
    pub const ZERO: Score = Score(0);

    fn checkmate_black() -> Score {
        Score::MAX.minus(1)
    }

    fn checkmate_white() -> Score {
        Score::MIN.plus(1)
    }

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
        sorted(board.with_color_to_move(BLACK).legal_moves(A1).unwrap()),
        sorted(cross_product(A1, vec![A2, B2, B1]))
    );

    assert_eq!(
        sorted(board.with_color_to_move(WHITE).legal_moves(C6).unwrap()),
        sorted(cross_product(C6, vec![C7, D7, D6, D5, C5, B5, B6, B7]))
    );

    assert_eq!(
        sorted(
            Board::empty()
                .place_piece(Piece(KING, BLACK), H8)
                .with_color_to_move(BLACK)
                .legal_moves(H8)
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
        sorted(board.legal_moves(F2).unwrap()),
        sorted(cross_product(F2, vec![G2, G1, E1, E3]))
    );
}

#[test]
fn test_king_cannot_move_into_check() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, BLACK), A1)
        .place_piece(Piece(KING, WHITE), H8)
        .place_piece(Piece(ROOK, WHITE), C2);

    assert_eq!(
        sorted(board.legal_moves(A1).unwrap()),
        sorted(vec![(A1, B1).into()])
    );
}

#[test]
fn test_king_cannot_move_into_check_by_other_king() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, BLACK), A1)
        .place_piece(Piece(KING, WHITE), C2);

    assert_eq!(
        sorted(board.legal_moves(A1).unwrap()),
        sorted(vec![(A1, A2).into()])
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
        .legal_moves(B8)
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
        crate::fen::parse("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 1 3")
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
fn test_castle_kingside_white_not_allowed_if_e1_under_attack() {
    init();

    // B4 bishop attacks E1
    let board =
        crate::fen::parse("rn1qk2r/pbpp1ppp/1p2pn2/8/1bPP4/3BPN2/PP3PPP/RNBQK2R w KQkq - 3 6")
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
fn test_castle_kingside_white_not_allowed_if_f1_under_attack() {
    init();

    // A6 bishop attacks F1
    let board =
        crate::fen::parse("rn1qkbnr/p1p1pppp/bp6/3p4/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
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
fn test_castle_kingside_white_not_allowed_if_g1_under_attack() {
    init();

    // G5 queen attacks E1
    let board =
        crate::fen::parse("rnb1kb1r/ppp2ppp/3pp3/4P1q1/2B5/4PN2/PPP2P1P/RNBQK2R w KQkq - 1 7")
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
        crate::fen::parse("r3kbnr/ppp1pppp/2nq4/3p1b2/3P1B2/2N5/PPPQPPPP/2KR1BNR b kq - 7 5")
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
fn test_castle_kingside_black() {
    init();

    let board =
        crate::fen::parse("r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b kq - 7 5")
            .unwrap();

    assert!(board
        .all_moves(Color::BLACK)
        .unwrap()
        .iter()
        .map(|mv| *mv)
        .collect::<Vec<Move>>()
        .contains(&Move::CastleKingside));

    let castled_board = board.make_move(Move::CastleKingside).unwrap();

    assert_eq!(
        castled_board,
        crate::fen::parse("r1bq1rk1/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 8 6")
            .unwrap()
    )
}

#[test]
fn test_castle_queenside_black() {
    init();

    let board =
        crate::fen::parse("r3kb1r/pppq1ppp/2np1n2/4pP2/3P4/2NBBN2/PPP2PPP/R2QK2R b KQkq - 2 7")
            .unwrap();

    assert!(board
        .all_moves(Color::BLACK)
        .unwrap()
        .iter()
        .map(|mv| *mv)
        .collect::<Vec<Move>>()
        .contains(&Move::CastleQueenside));

    let castled_board = board.make_move(Move::CastleQueenside).unwrap();

    assert_eq!(
        castled_board,
        crate::fen::parse("2kr1b1r/pppq1ppp/2np1n2/4pP2/3P4/2NBBN2/PPP2PPP/R2QK2R w KQ - 3 8")
            .unwrap()
    )
}

#[test]
fn test_cannot_castle_after_moving_white_king() {
    init();

    let board = crate::fen::parse(
        "r3k2r/ppp2ppp/2nq1n2/1B1pp1B1/1b1PP1b1/2NQ1N2/PPP2PPP/R3K2R w KQkq - 10 8",
    )
    .unwrap();

    let moved_board = board
        .make_move((E1, D1).into())
        .unwrap()
        .make_move((A7, A6).into())
        .unwrap()
        .make_move((D1, E1).into())
        .unwrap();

    assert!(!moved_board.can_castle_kingside(WHITE).unwrap());
    assert!(moved_board.can_castle_kingside(BLACK).unwrap());
    assert!(!moved_board.can_castle_queenside(WHITE).unwrap());
    assert!(moved_board.can_castle_queenside(BLACK).unwrap());
}

#[test]
fn test_cannot_castle_after_moving_black_king() {
    init();

    let board = crate::fen::parse(
        "r3k2r/ppp2ppp/2nq1n2/1B1pp1B1/1b1PP1b1/2NQ1N1P/PPP2PP1/R3K2R b KQkq - 0 8",
    )
    .unwrap();

    let moved_board = board
        .make_move((E8, E7).into())
        .unwrap()
        .make_move((A2, A3).into())
        .unwrap()
        .make_move((E7, E8).into())
        .unwrap();

    assert!(moved_board.can_castle_kingside(WHITE).unwrap());
    assert!(!moved_board.can_castle_kingside(BLACK).unwrap());
    assert!(moved_board.can_castle_queenside(WHITE).unwrap());
    assert!(!moved_board.can_castle_queenside(BLACK).unwrap());
}

#[test]
fn test_cannot_castle_queenside_after_moving_a1_rook() {
    init();

    let board = crate::fen::parse(
        "r3k2r/ppp2ppp/2nq1n2/1B1pp1B1/1b1PP1b1/2NQ1N2/PPP2PPP/R3K2R w KQkq - 10 8",
    )
    .unwrap();

    let moved_board = board
        .make_move((A1, B1).into())
        .unwrap()
        .make_move((A7, A6).into())
        .unwrap()
        .make_move((B1, A1).into())
        .unwrap();

    assert!(moved_board.can_castle_kingside(WHITE).unwrap());
    assert!(moved_board.can_castle_kingside(BLACK).unwrap());
    assert!(!moved_board.can_castle_queenside(WHITE).unwrap());
    assert!(moved_board.can_castle_queenside(BLACK).unwrap());
}

#[test]
fn test_cannot_castle_queenside_after_moving_a8_rook() {
    init();

    let board =
        crate::fen::parse("r3k2r/ppp2ppp/2nq1n2/1B1pN1B1/1b1PP1b1/2NQ4/PPP2PPP/R3K2R b KQkq - 0 8")
            .unwrap();

    let moved_board = board
        .make_move((A8, D8).into())
        .unwrap()
        .make_move((F2, F3).into())
        .unwrap()
        .make_move((D8, A8).into())
        .unwrap();

    assert!(moved_board.can_castle_kingside(WHITE).unwrap());
    assert!(moved_board.can_castle_kingside(BLACK).unwrap());
    assert!(moved_board.can_castle_queenside(WHITE).unwrap());
    assert!(!moved_board.can_castle_queenside(BLACK).unwrap());
}

#[test]
fn test_cannot_castle_kingside_after_moving_h8_rook() {
    init();

    let board = crate::fen::parse(
        "r3k2r/ppp2ppp/2nq1n2/1B1pp1B1/1b1PP1b1/2NQ1N1P/PPP2PP1/R3K2R b KQkq - 0 8",
    )
    .unwrap();

    let moved_board = board
        .make_move((H8, G8).into())
        .unwrap()
        .make_move((A2, A3).into())
        .unwrap()
        .make_move((G8, H8).into())
        .unwrap();

    assert!(moved_board.can_castle_kingside(WHITE).unwrap());
    assert!(!moved_board.can_castle_kingside(BLACK).unwrap());
    assert!(moved_board.can_castle_queenside(WHITE).unwrap());
    assert!(moved_board.can_castle_queenside(BLACK).unwrap());
}

#[test]
fn test_cannot_castle_kingside_after_moving_h1_rook() {
    init();

    let board = crate::fen::parse(
        "r3k2r/ppp2ppp/2nq1n2/1B1pp1B1/1b1PP1b1/2NQ1N2/PPP2PPP/R3K2R w KQkq - 10 8",
    )
    .unwrap();

    let moved_board = board
        .make_move((H1, F1).into())
        .unwrap()
        .make_move((A7, A6).into())
        .unwrap()
        .make_move((F1, H1).into())
        .unwrap();

    assert!(!moved_board.can_castle_kingside(WHITE).unwrap());
    assert!(moved_board.can_castle_kingside(BLACK).unwrap());
    assert!(moved_board.can_castle_queenside(WHITE).unwrap());
    assert!(moved_board.can_castle_queenside(BLACK).unwrap());
}

#[test]
fn test_rook_free_movement() {
    let board = Board::empty()
        .place_piece(Piece(KING, WHITE), D3)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(ROOK, WHITE), C6);

    assert_eq!(
        sorted(board.legal_moves(A1).unwrap()),
        sorted(cross_product(
            A1,
            vec![
                A2, A3, A4, A5, A6, A7, A8, // rank moves (up-down)
                B1, C1, D1, E1, F1, G1, H1 // file moves (side-to-side)
            ]
        ))
    );

    assert_eq!(
        sorted(board.legal_moves(C6).unwrap()),
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
    let board = Board::empty()
        .place_piece(Piece(KING, WHITE), C1)
        .place_piece(Piece(KING, BLACK), F5)
        .place_piece(Piece(ROOK, WHITE), A8);

    assert_eq!(
        sorted(board.legal_moves(A8).unwrap()),
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
        .place_piece(Piece(KING, WHITE), E2)
        .place_piece(Piece(PAWN, WHITE), E6)
        .place_piece(Piece(PAWN, WHITE), A5)
        .place_piece(Piece(PAWN, WHITE), G5)
        .place_piece(Piece(KING, BLACK), H8);

    assert_eq!(
        sorted(board.legal_moves(E5).unwrap()),
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
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(ROOK, WHITE), E5)
        .place_piece(Piece(PAWN, BLACK), E2)
        .place_piece(Piece(PAWN, BLACK), E6)
        .place_piece(Piece(PAWN, BLACK), A5)
        .place_piece(Piece(PAWN, BLACK), G5);

    assert_eq!(
        sorted(board.legal_moves(E5).unwrap()),
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
        .with_color_to_move(BLACK)
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(ROOK, WHITE), B2)
        .place_piece(Piece(KING, BLACK), A6);

    assert!(board.checkmate(BLACK).unwrap());
}

#[test]
fn test_can_escape_checkmate() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
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
fn test_avoid_stalemate() {
    init();
    let board = crate::fen::parse("6k1/6p1/4K1P1/7r/8/3q4/8/8 b - - 3 62").unwrap();

    let (mv, _) = board.find_next_move(4).unwrap().unwrap();
    // Would be stalemate:
    assert_ne!(mv, Move::Single { from: D3, to: D8 });

    Puzzle::new(board)
        .should_find_move(D3, D2)
        .respond_with(E6, E7)
        .should_find_move(H5, E5)
        .should_be_checkmate();
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

// #[test]
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
    let board = Board::empty()
        .place_piece(Piece(PAWN, WHITE), A2)
        .place_piece(Piece(KING, WHITE), C1)
        .place_piece(Piece(KING, BLACK), H8);

    assert_eq!(
        sorted(board.legal_moves(A2).unwrap()),
        sorted(cross_product(A2, vec![A3, A4,]))
    );
}

#[test]
fn test_pawn_movement_from_start_rank_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), C7)
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8);

    assert_eq!(
        sorted(board.legal_moves(C7).unwrap()),
        sorted(cross_product(C7, vec![C6, C5,]))
    );
}

#[test]
fn test_pawn_movement_blocked_from_start_rank_white() {
    init();
    let board1 = Board::empty()
        .place_piece(Piece(KING, WHITE), H8)
        .place_piece(Piece(PAWN, WHITE), A2)
        .place_piece(Piece(KING, BLACK), A4);

    assert_eq!(
        sorted(board1.legal_moves(A2).unwrap()),
        sorted(vec![(A2, A3).into()])
    );

    let board2 = Board::empty()
        .place_piece(Piece(PAWN, WHITE), A2)
        .place_piece(Piece(KING, BLACK), A3)
        .place_piece(Piece(KING, WHITE), H8);

    assert_eq!(sorted(board2.legal_moves(A2).unwrap()), Vec::new());
}

#[test]
fn test_pawn_movement_blocked_from_start_rank_black() {
    init();
    let board1 = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), A7)
        .place_piece(Piece(KING, WHITE), A5)
        .place_piece(Piece(KING, BLACK), F6);

    assert_eq!(
        sorted(board1.legal_moves(A7).unwrap()),
        sorted(vec![(A7, A6).into()])
    );

    let board2 = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), A7)
        .place_piece(Piece(KING, WHITE), A6)
        .place_piece(Piece(KING, BLACK), C4);

    assert_eq!(sorted(board2.legal_moves(A7).unwrap()), Vec::new());
}

#[test]
fn test_pawn_movement_from_middle_board_white() {
    init();
    let board = Board::empty()
        .place_piece(Piece(PAWN, WHITE), H3)
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8);

    assert_eq!(
        sorted(board.legal_moves(H3).unwrap()),
        sorted(vec![(H3, H4).into()])
    );
}

#[test]
fn test_pawn_movement_from_middle_board_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(PAWN, BLACK), D4)
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8);

    assert_eq!(
        sorted(board.legal_moves(D4).unwrap()),
        sorted(vec![(D4, D3).into()])
    );
}

#[test]
fn test_pawn_capture_white() {
    init();
    let board = Board::empty()
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(PAWN, WHITE), F6)
        .place_piece(Piece(ROOK, BLACK), E7)
        .place_piece(Piece(ROOK, BLACK), G7);

    assert_eq!(
        sorted(board.legal_moves(F6).unwrap()),
        sorted(cross_product(F6, vec![E7, F7, G7]))
    );
}

#[test]
fn test_pawn_capture_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(PAWN, BLACK), C4)
        .place_piece(Piece(ROOK, WHITE), B3)
        .place_piece(Piece(ROOK, WHITE), D3);

    assert_eq!(
        sorted(board.legal_moves(C4).unwrap()),
        sorted(cross_product(C4, vec![B3, C3, D3]))
    );
}

#[test]
fn test_pawn_cannot_capture_own_pieces_white() {
    init();
    let board = Board::empty()
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(PAWN, WHITE), F6)
        .place_piece(Piece(ROOK, WHITE), E7)
        .place_piece(Piece(ROOK, WHITE), G7);

    assert_eq!(
        sorted(board.legal_moves(F6).unwrap()),
        sorted(vec![(F6, F7).into()])
    );
}

#[test]
fn test_pawn_cannot_capture_own_pieces_black() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(PAWN, BLACK), C4)
        .place_piece(Piece(ROOK, BLACK), B3)
        .place_piece(Piece(ROOK, BLACK), D3);

    assert_eq!(
        sorted(board.legal_moves(C4).unwrap()),
        sorted(vec![(C4, C3).into()])
    );
}

#[test]
fn test_pawn_cannot_capture_around_edge_of_board() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), C3)
        .place_piece(Piece(KING, BLACK), E4)
        .place_piece(Piece(PAWN, WHITE), A6)
        .place_piece(Piece(ROOK, BLACK), H8)
        .place_piece(Piece(PAWN, WHITE), H3)
        .place_piece(Piece(ROOK, BLACK), A5);

    assert_eq!(
        sorted(board.legal_moves(A6).unwrap()),
        sorted(vec![(A6, A7).into()])
    );

    assert_eq!(
        sorted(board.legal_moves(H3).unwrap()),
        sorted(vec![(H3, H4).into()])
    );
}

#[test]
fn test_pawn_promote_to_queen() {
    init();

    let board = crate::fen::parse("5k2/1P6/8/8/2B5/K5Pp/1P5P/8 w - - 1 55").unwrap();

    assert_eq!(
        sorted(board.legal_moves(B7).unwrap()),
        sorted(vec![
            Move::Promote {
                from: B7,
                to: B8,
                piece: Piece(QUEEN, Color::WHITE)
            },
            Move::Promote {
                from: B7,
                to: B8,
                piece: Piece(ROOK, Color::WHITE)
            },
            Move::Promote {
                from: B7,
                to: B8,
                piece: Piece(BISHOP, Color::WHITE)
            },
            Move::Promote {
                from: B7,
                to: B8,
                piece: Piece(KNIGHT, Color::WHITE)
            }
        ])
    );

    // TODO: this should be the best move when searching greater depths
    let (best_move, _) = board.find_next_move(2).unwrap().unwrap();

    assert_eq!(
        best_move,
        Move::Promote {
            from: B7,
            to: B8,
            piece: Piece(QUEEN, Color::WHITE)
        }
    );

    let moved_board = board.make_move(best_move).unwrap();

    assert_eq!(
        moved_board,
        crate::fen::parse("1Q3k2/8/8/8/2B5/K5Pp/1P5P/8 b - - 0 55").unwrap()
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
//         sorted(board.legal_moves(E5).unwrap()),
//         sorted(vec![E6, D6])
//     );
// }

#[test]
fn test_knight_moves() {
    init();
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(KNIGHT, BLACK), D4);

    assert_eq!(
        sorted(board.legal_moves(D4).unwrap()),
        sorted(cross_product(D4, vec![E6, F5, F3, E2, C2, B3, B5, C6,]))
    );
}

#[test]
fn test_bishop_moves() {
    init();
    let board = Board::empty()
        .with_color_to_move(WHITE)
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(BISHOP, WHITE), C5);

    assert_eq!(
        sorted(board.legal_moves(C5).unwrap()),
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
        .place_piece(Piece(KING, WHITE), A1)
        .place_piece(Piece(KING, BLACK), H8)
        .place_piece(Piece(QUEEN, WHITE), C5);

    assert_eq!(
        sorted(board.legal_moves(C5).unwrap()),
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

#[test]
fn test_record_en_passant_target() {
    let board =
        crate::fen::parse("rnbqkbnr/1ppppppp/p7/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2").unwrap();

    assert_eq!(
        board
            .make_move(Move::Single { from: D7, to: D5 })
            .unwrap()
            .en_passant_target,
        Some(D6)
    );

    assert_eq!(
        board
            .make_move(Move::Single { from: F7, to: F5 })
            .unwrap()
            .en_passant_target,
        Some(F6)
    );
}

#[test]
fn test_clear_en_passant_target() {
    let board =
        crate::fen::parse("rnbqkbnr/1ppppppp/p7/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2").unwrap();

    // En passant target is cleared after opponent makes a move
    assert_eq!(
        board
            .make_move(Move::Single { from: F7, to: F5 })
            .unwrap()
            .make_move(Move::Single { from: D2, to: D3 })
            .unwrap()
            .en_passant_target,
        None
    );
}

#[test]
fn test_en_passant_capture_white() {
    let board =
        crate::fen::parse("rnbqkbnr/2ppp1pp/1p6/p3PpP1/8/8/PPPP1P1P/RNBQKBNR w KQkq f6 0 5")
            .unwrap();

    assert!(board
        .legal_moves(G5)
        .unwrap()
        .contains(&Move::Single { from: G5, to: F6 }));

    assert!(board
        .legal_moves(E5)
        .unwrap()
        .contains(&Move::Single { from: E5, to: F6 }));

    let captured_board = board.make_move(Move::Single { from: E5, to: F6 }).unwrap();

    assert!(captured_board.is_empty(E5.index())); // capturing pawn has moved
    assert!(captured_board.is_empty(F5.index())); // captured pawn is gone
    assert!(captured_board.piece_on_square(F6) == Some(Piece(PAWN, WHITE))); // capturer moved to en passant target
}

#[test]
fn test_en_passant_capture_black() {
    let board =
        crate::fen::parse("rnbqkbnr/1p1ppppp/8/8/pPp2PPP/8/P1PPP3/RNBQKBNR b KQkq b3 0 5").unwrap();

    assert!(board
        .legal_moves(A4)
        .unwrap()
        .contains(&Move::Single { from: A4, to: B3 }));

    assert!(board
        .legal_moves(C4)
        .unwrap()
        .contains(&Move::Single { from: C4, to: B3 }));

    let captured_board = board.make_move(Move::Single { from: A4, to: B3 }).unwrap();

    assert!(captured_board.is_empty(A4.index())); // capturing pawn has moved
    assert!(captured_board.is_empty(B4.index())); // captured pawn is gone
    assert!(captured_board.piece_on_square(B3) == Some(Piece(PAWN, BLACK))); // capturer moved to en passant target
}

#[test]
fn test_foo() {
    let board = crate::fen::parse("5rk1/3n2pp/2p1p3/5pP1/1PPP4/8/5PP1/R5K1 b - - 0 26").unwrap();

    board.find_next_move(4).unwrap();
}
