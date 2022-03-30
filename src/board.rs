use crate::bitboard::*;
use crate::color::*;
use crate::constants::*;
use crate::error::BoardError;
use crate::error::BoardError::*;
use crate::evaluate::*;
use crate::file::*;
use crate::mv::*;
use crate::rank::*;
use crate::square::*;
use crate::traversal_path::TraversalPath;
use crate::vector::*;
use log::{debug, info};
use rayon::prelude::*;
use std::cmp;
use std::fmt;
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

pub type Result<T> = std::result::Result<T, BoardError>;

pub use PieceEnum::*;

pub type History = Vec<(Move, Color)>;
pub type PV = Vec<Move>;

/*
 * Convert a principal variation into a full history containing the color of
 * each move.
 */
fn full_history(pv: PV, start_color: Color) -> History {
    pv.into_iter()
        .zip(
            std::iter::repeat([start_color, start_color.opposite()])
                .flat_map(|array| array.into_iter()),
        )
        .collect()
}

struct HistoryDisplay<'a>(&'a History);

impl fmt::Display for HistoryDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (mv, _)) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", mv.to_string())?;
        }

        Ok(())
    }
}

pub fn move_generator(board: &Board, last_move: Option<Move>) -> impl Iterator<Item = Move> + '_ {
    GenIter(move || {
        let color = board.color_to_move;

        // recaptures

        let last_destination = last_move.and_then(|mv| match mv {
            Move::Single { to, .. } | Move::Promote { to, .. } => Some(to),
            _ => None,
        });

        if let Some(last_destination) = last_destination {
            for recapturing_mv in
                board.attacking_moves_by_color(last_destination, board.color_to_move)
            {
                yield recapturing_mv
            }
        }

        for mv in board.all_pawn_captures(color) {
            yield mv
        }

        fn unpack_attacking_moves(
            from: Square,
            attacks: Bitboard,
            defenders: Bitboard,
        ) -> impl Iterator<Item = Move> {
            (attacks & defenders)
                .squares()
                .map(move |to| Move::Single { from, to })
        }

        let mut defenders = board.presence_for(color.opposite()).all;
        // unset the last defender bit, we already yielded those moves as part of recapturing above
        if let Some(last_destination) = last_destination {
            defenders ^= bitboard![last_destination];
        }

        for (from, attacks) in board.all_bishop_attacks(color) {
            for mv in unpack_attacking_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for (from, attacks) in board.all_rook_attacks(color) {
            for mv in unpack_attacking_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for (from, attacks) in board.all_knight_attacks(color) {
            for mv in unpack_attacking_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for (from, attacks) in board.all_queen_attacks(color) {
            for mv in unpack_attacking_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for (from, attacks) in board.all_king_attacks(color) {
            for mv in unpack_attacking_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for mv in board.en_passant_captures(color) {
            yield mv
        }

        // castling

        if board.can_castle_kingside(color) {
            yield Move::CastleKingside;
        }

        if board.can_castle_queenside(color) {
            yield Move::CastleQueenside;
        }

        // quiet moves

        fn unpack_quiet_moves(
            from: Square,
            attacks: Bitboard,
            defenders: Bitboard,
        ) -> impl Iterator<Item = Move> {
            (attacks & !defenders)
                .squares()
                .map(move |to| Move::Single { from, to })
        }

        let defenders = board.presence_for(color.opposite()).all;

        for (from, attacks) in board.all_bishop_attacks(color) {
            for mv in unpack_quiet_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for (from, attacks) in board.all_rook_attacks(color) {
            for mv in unpack_quiet_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for (from, attacks) in board.all_knight_attacks(color) {
            for mv in unpack_quiet_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for (from, attacks) in board.all_queen_attacks(color) {
            for mv in unpack_quiet_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for (from, attacks) in board.all_king_attacks(color) {
            for mv in unpack_quiet_moves(from, attacks, defenders) {
                yield mv
            }
        }

        for mv in board.all_pawn_advances(color) {
            yield mv
        }
    })
}

/*
 * Convert a generator into an iterator.
 *
 * Borrowed from the `gen_iter` crate: https://docs.rs/gen-iter/latest/gen_iter/
 */
#[derive(Copy, Clone, Debug)]
pub struct GenIter<T>(pub T)
where
    T: Generator<Return = ()> + Unpin;

impl<T> Iterator for GenIter<T>
where
    T: Generator<Return = ()> + Unpin,
{
    type Item = T::Yield;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match Pin::new(&mut self.0).resume(()) {
            GeneratorState::Yielded(n) => Some(n),
            GeneratorState::Complete(()) => None,
        }
    }
}

/*
 * TODO:
 * - ingest lichess puzzles in a test suite
 * - prioritize king safety
 * - PERFT
 * - better move ordering
 *
 * See https://github.com/AndyGrant/Ethereal/blob/master/src/movegen.c
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Copy, Hash)]
pub struct Piece(pub PieceEnum, pub Color);

impl Piece {
    fn piece(&self) -> PieceEnum {
        self.0
    }

    fn color(&self) -> Color {
        self.1
    }
}

pub const ALL_PIECES: [PieceEnum; 6] = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Copy, Hash)]
pub enum PieceEnum {
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
}

impl fmt::Display for PieceEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            PAWN => 'p',
            KNIGHT => 'n',
            BISHOP => 'b',
            ROOK => 'r',
            QUEEN => 'q',
            KING => 'k',
        };

        write!(f, "{}", symbol)
    }
}

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

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct Presence {
    pub color: Color,
    pub pawn: Bitboard,
    pub knight: Bitboard,
    pub bishop: Bitboard,
    pub rook: Bitboard,
    pub queen: Bitboard,
    pub king: Bitboard,
    pub all: Bitboard,
}

impl Presence {
    pub fn empty(color: Color) -> Self {
        Self {
            color: color,
            pawn: Bitboard::empty(),
            knight: Bitboard::empty(),
            bishop: Bitboard::empty(),
            rook: Bitboard::empty(),
            queen: Bitboard::empty(),
            king: Bitboard::empty(),
            all: Bitboard::empty(),
        }
    }

    pub fn for_piece(&self, piece: PieceEnum) -> &Bitboard {
        match piece {
            PAWN => &self.pawn,
            KNIGHT => &self.knight,
            BISHOP => &self.bishop,
            ROOK => &self.rook,
            QUEEN => &self.queen,
            KING => &self.king,
        }
    }

    pub fn for_piece_mut(&mut self, piece: PieceEnum) -> &mut Bitboard {
        match piece {
            PAWN => &mut self.pawn,
            KNIGHT => &mut self.knight,
            BISHOP => &mut self.bishop,
            ROOK => &mut self.rook,
            QUEEN => &mut self.queen,
            KING => &mut self.king,
        }
    }
}

fn compute_attacks(
    from: Square,
    rays: &[Bitboard; 64],
    same_color: Bitboard,
    other_color: Bitboard,
    unchecked_bitscan: fn(Bitboard) -> Square,
    backstop: Square,
) -> Bitboard {
    let intersections = rays[from.index()] & (same_color | other_color);

    // Because the backstop is included above, we know that at least one bit
    // will be set on the board, hence can use the unchecked bitscans

    let blocker = unchecked_bitscan(intersections | bitboard![backstop]);

    !same_color & rays[from.index()] & !rays[blocker.index()]
}

// pinned pieces are all the defenders which stand between an attacking slider and the defenders king

fn pinned_pieces(attackers: Presence, defenders: Presence) -> Bitboard {
    Bitboard::empty()
}

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct Board {
    pub color_to_move: Color,
    en_passant_target: Option<Square>,
    // # of moves since last capture or pawn advance. For enforcing the 50-move rule.
    halfmove_clock: u16,
    // Move #, incremented after Black plays
    fullmove_clock: u16,
    can_castle: CastleRights,

    pub presence_white: Presence,
    pub presence_black: Presence,
}

impl Board {
    pub fn empty() -> Self {
        Board {
            color_to_move: WHITE,
            en_passant_target: None,
            halfmove_clock: 0,
            fullmove_clock: 1,
            can_castle: CastleRights::none(),
            presence_white: Presence::empty(WHITE),
            presence_black: Presence::empty(BLACK),
        }
    }

    fn presence_for(&self, color: Color) -> &Presence {
        match color {
            WHITE => &self.presence_white,
            BLACK => &self.presence_black,
        }
    }

    fn presence_for_mut(&mut self, color: Color) -> &mut Presence {
        match color {
            WHITE => &mut self.presence_white,
            BLACK => &mut self.presence_black,
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

        let ours = new.presence_for_mut(piece.color());
        ours.all |= bitboard![on];
        *ours.for_piece_mut(piece.piece()) |= bitboard![on];

        new
    }

    // Incremental move. Updates all internal state for the board: mailbox,
    // bitboards, clocks.
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

        debug_assert!((self.presence_white.all & self.presence_black.all).is_empty());
        Ok(new)
    }

    fn is_pawn_advance(&self, mv: Move) -> Result<bool> {
        match mv {
            Move::Single { from, .. } => Ok((bitboard![from]
                & (self.presence_white.pawn | self.presence_black.pawn))
                .non_empty()),
            Move::Promote { .. } => Ok(true),
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
            } => {
                if (self.presence_for(self.color_to_move.opposite()).all & bitboard!(to))
                    .non_empty()
                {
                    Ok(true)
                } else if (self.presence_for(self.color_to_move).all & bitboard!(to)).non_empty() {
                    Err(IllegalMove(
                        "Trying to capture piece of same color".to_string(),
                    ))
                } else {
                    Ok(false)
                }
            }
            Move::CastleKingside => Ok(false),
            Move::CastleQueenside => Ok(false),
        }
    }

    fn castle_kingside(&mut self, color: Color) -> Result<()> {
        if !self.can_castle_kingside(color) {
            return Err(IllegalCastle);
        }

        if color == Color::WHITE {
            let king_mask = bitboard!(E1, G1);
            let rook_mask = bitboard!(F1, H1);

            self.presence_white.king ^= king_mask;
            self.presence_white.rook ^= rook_mask;

            self.presence_white.all ^= king_mask;
            self.presence_white.all ^= rook_mask;

            self.can_castle.kingside_white = false;
            self.can_castle.queenside_white = false;
        } else {
            let king_mask = bitboard!(E8, G8);
            let rook_mask = bitboard!(F8, H8);

            self.presence_black.king ^= king_mask;
            self.presence_black.rook ^= rook_mask;

            self.presence_black.all ^= king_mask;
            self.presence_black.all ^= rook_mask;

            self.can_castle.kingside_black = false;
            self.can_castle.queenside_black = false;
        }

        Ok(())
    }

    fn castle_queenside(&mut self, color: Color) -> Result<()> {
        if !self.can_castle_queenside(color) {
            return Err(IllegalCastle);
        }

        if color == Color::WHITE {
            let king_mask = bitboard!(C1, E1);
            let rook_mask = bitboard!(A1, D1);

            self.presence_white.king ^= king_mask;
            self.presence_white.rook ^= rook_mask;

            self.presence_white.all ^= king_mask;
            self.presence_white.all ^= rook_mask;

            self.can_castle.kingside_white = false;
            self.can_castle.queenside_white = false;
        } else {
            let king_mask = bitboard!(C8, E8);
            let rook_mask = bitboard!(A8, D8);

            self.presence_black.king ^= king_mask;
            self.presence_black.rook ^= rook_mask;

            self.presence_black.all ^= king_mask;
            self.presence_black.all ^= rook_mask;

            self.can_castle.kingside_black = false;
            self.can_castle.queenside_black = false;
        }

        Ok(())
    }

    fn can_castle_kingside(&self, color: Color) -> bool {
        let allowed = match color {
            Color::WHITE => {
                self.can_castle.kingside_white
                    && self.is_empty(F1)
                    && self.is_empty(G1)
                    && !self.attacked_by_color(E1, color.opposite())
                    && !self.attacked_by_color(F1, color.opposite())
                    && !self.attacked_by_color(G1, color.opposite())
            }
            Color::BLACK => {
                self.can_castle.kingside_black
                    && self.is_empty(F8)
                    && self.is_empty(G8)
                    && !self.attacked_by_color(E8, color.opposite())
                    && !self.attacked_by_color(F8, color.opposite())
                    && !self.attacked_by_color(G8, color.opposite())
            }
        };

        allowed
    }

    fn can_castle_queenside(&self, color: Color) -> bool {
        let allowed = match color {
            Color::WHITE => {
                self.can_castle.queenside_white
                    && self.is_empty(B1)
                    && self.is_empty(C1)
                    && self.is_empty(D1)
                    && !self.attacked_by_color(C1, color.opposite())
                    && !self.attacked_by_color(D1, color.opposite())
                    && !self.attacked_by_color(E1, color.opposite())
            }
            Color::BLACK => {
                self.can_castle.queenside_black
                    && self.is_empty(B8)
                    && self.is_empty(C8)
                    && self.is_empty(D8)
                    && !self.attacked_by_color(C8, color.opposite())
                    && !self.attacked_by_color(D8, color.opposite())
                    && !self.attacked_by_color(E8, color.opposite())
            }
        };

        allowed
    }

    fn move_piece(&mut self, from: Square, to: Square) -> Result<()> {
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
                    .map(|target| self.is_empty(target))
                    .unwrap_or(false)
            {
                // The captured piece is one rank different than the target square
                let captured_square = Square::from_index(match self.color_to_move {
                    WHITE => to.index() - N_FILES,
                    BLACK => to.index() + N_FILES,
                });

                assert!(!self.is_empty(captured_square));

                let theirs = self.presence_for_mut(self.color_to_move.opposite());
                let mask = bitboard!(captured_square);
                theirs.pawn ^= mask;
                theirs.all ^= mask;
            }

            // Update castling - moving off original squares
            if piece == ROOK && color == WHITE && from == A1 {
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

            // Update castling - capturing rook
            if let Some(captured) = self.piece_on_square(to) {
                self.update_castle_rights(captured, to)
            }

            // update bitboards
            let captured = self.piece_on_square(to);
            let ours = self.presence_for_mut(color);

            let from_mask = bitboard!(from);
            let to_mask = bitboard!(to);

            *ours.for_piece_mut(piece) ^= from_mask;
            ours.all ^= from_mask;

            *ours.for_piece_mut(piece) ^= to_mask;
            ours.all ^= to_mask;

            if let Some(captured) = captured {
                let theirs = self.presence_for_mut(captured.color());
                *theirs.for_piece_mut(captured.piece()) ^= to_mask;
                theirs.all ^= to_mask;
            }
        } else {
            return Err(NoPieceOnFromSquare(from));
        }

        // TODO: verify that move is valid.

        Ok(())
    }

    fn promote_pawn(&mut self, from: Square, to: Square, promote_to: Piece) -> Result<()> {
        // TODO: check that this is actually a promoting move. Impose type constraints?

        assert!(promote_to.color() == self.color_to_move);

        if !self.contains_piece(from, PAWN, promote_to.color()) {
            return Err(NoPieceOnFromSquare(from));
        }

        // save this here before we start mucking with the board

        let captured = self.piece_on_square(to);

        // update bitboards

        let ours = self.presence_for_mut(promote_to.color());
        let pawn_mask = bitboard!(from);
        let promoted_mask = bitboard!(to);

        ours.pawn ^= pawn_mask;
        ours.all ^= pawn_mask;

        *ours.for_piece_mut(promote_to.piece()) |= promoted_mask;
        ours.all |= promoted_mask;

        // in case of capture, remove the opponents piece

        if let Some(captured) = captured {
            let theirs = self.presence_for_mut(captured.color());
            *theirs.for_piece_mut(captured.piece()) ^= promoted_mask;
            theirs.all ^= promoted_mask;

            self.update_castle_rights(captured, to);
        }

        Ok(())
    }

    // Update castling rights after a capture

    fn update_castle_rights(&mut self, captured: Piece, on: Square) -> () {
        if captured.piece() == ROOK {
            if captured.color() == WHITE && on == A1 {
                self.can_castle.queenside_white = false
            } else if captured.color() == BLACK && on == A8 {
                self.can_castle.queenside_black = false
            } else if captured.color() == WHITE && on == H1 {
                self.can_castle.kingside_white = false
            } else if captured.color() == BLACK && on == H8 {
                self.can_castle.kingside_black = false
            }
        }
    }

    fn is_empty(&self, square: Square) -> bool {
        (bitboard![square] & (self.presence_white.all | self.presence_black.all)).is_empty()
    }

    pub fn contains_piece(&self, square: Square, piece: PieceEnum, color: Color) -> bool {
        (bitboard![square] & *self.presence_for(color).for_piece(piece)).non_empty()
    }

    // Note: this is inefficient because it needs to iterate all the piece
    // bitboards. If possible look directly at the bitboard of interest instead.

    pub fn piece_on_square(&self, square: Square) -> Option<Piece> {
        let side = if (bitboard![square] & self.presence_white.all).non_empty() {
            &self.presence_white
        } else if (bitboard![square] & self.presence_black.all).non_empty() {
            &self.presence_black
        } else {
            return None;
        };

        for piece in ALL_PIECES.into_iter() {
            if (bitboard![square] & *side.for_piece(piece)).non_empty() {
                return Some(Piece(piece, side.color));
            }
        }

        return None;
    }

    pub fn is_in_check(&self, color: Color) -> Result<bool> {
        let mut king_squares = self.presence_for(color).king.squares();

        let king_square = king_squares
            .next()
            .ok_or_else(|| IllegalState(format!("Board is missing KING of color {}", color)))?;

        if king_squares.next().is_some() {
            return Err(IllegalState(format!(
                "Board has more than on KING of color {}",
                color
            )));
        }

        Ok(self.attacked_by_color(king_square, color.opposite()))
    }

    // Attacking moves are a subset of other moves.

    fn attacked_by_color(&self, target_square: Square, color: Color) -> bool {
        self.attacking_moves_by_color(target_square, color)
            .next()
            .is_some()
    }

    // TODO: include en passant here?
    fn attacking_moves_by_color<'a>(
        &'a self,
        target_square: Square,
        color: Color,
    ) -> impl Iterator<Item = Move> + 'a {
        // Observation: computing attackers of a square is the inverse operation
        // as computing attacks *from* the target square, and computing from the
        // target square requires many fewer operations.

        // Observation: queen attacks can be computed at the same time as rook
        // and bishop attacks, just need to AND with the queen bitboard as well.

        GenIter(move || {
            let attackers = self.presence_for(color);
            let to = target_square;

            let bishop_attacks = self
                ._bishop_attacks(target_square, &Piece(BISHOP, color.opposite()))
                & (attackers.bishop | attackers.queen);
            for from in bishop_attacks.squares() {
                yield Move::Single { from, to }
            }

            let rook_attacks = self._rook_attacks(target_square, &Piece(ROOK, color.opposite()))
                & (attackers.rook | attackers.queen);
            for from in rook_attacks.squares() {
                yield Move::Single { from, to }
            }

            let knight_attacks = self
                ._knight_attacks(target_square, &Piece(KNIGHT, color.opposite()))
                & attackers.knight;
            for from in knight_attacks.squares() {
                yield Move::Single { from, to }
            }

            let king_attacks =
                self._king_attacks(target_square, &Piece(KING, color.opposite())) & attackers.king;
            for from in king_attacks.squares() {
                yield Move::Single { from, to }
            }

            let pawn_attacks =
                self._pawn_attacks(target_square, &Piece(PAWN, color.opposite())) & attackers.pawn;
            for from in pawn_attacks.squares() {
                yield Move::Single { from, to }
            }
        })
    }

    // Return all candidate moves for single pieces, allowing illegal moves
    // (e.g., that move the king into check)
    fn candidate_moves(&self, history: &History) -> impl Iterator<Item = Move> + '_ {
        return move_generator(self, history.last().map(|(mv, _)| *mv));
    }

    // // Return all legal moves for the given square, filtering out those that result in
    // // illegal states
    // //
    // // NOTE: do not use this. Inefficient, better to compare against the already
    // // moved board in find_next_move.
    #[cfg(test)]
    fn legal_moves(&self, from: Square) -> Result<Vec<Move>> {
        // TODO assert that piece on square is color to move

        let mut moves = Vec::new();

        let Piece(_, color) = self
            .piece_on_square(from)
            .ok_or(NoPieceOnFromSquare(from))?;

        if color != self.color_to_move {
            return Err(BoardError::IllegalMove(format!("Not {}'s turn", color)));
        }

        for mv in self.candidate_moves(&mut Vec::new()) {
            match mv {
                Move::Promote {
                    from: source,
                    to: _,
                    piece: _,
                }
                | Move::Single {
                    from: source,
                    to: _,
                } if from == source => (),
                _ => continue,
            }

            let moved_board = self.make_move(mv)?;
            // Cannot move into check
            if !moved_board.is_in_check(self.color_to_move)? {
                moves.push(mv)
            }
        }

        Ok(moves)
    }

    // Return all moves possible for the given color, including castling and promotion
    pub fn all_moves(&self) -> impl Iterator<Item = Move> + '_ {
        self.candidate_moves(&mut Vec::new())
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

    pub fn plus_vector_scaled<'a>(
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

    // Expand all_pawn_attacks to valid moves
    fn all_pawn_captures<'a>(&'a self, color: Color) -> impl Iterator<Item = Move> + 'a {
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

        self.all_pawn_attacks(color)
            .flat_map(move |(from, bitboard)| {
                // Only look at attacked squares which contain opponent's pieces
                let bitboard = bitboard & self.presence_for(color.opposite()).all;

                // Only one of these is going to set anything
                let promoting = bitboard & (RANK_1 | RANK_8);
                let not_promoting = bitboard & !(RANK_1 | RANK_8);

                // Expand those attacked squares into concrete moves

                // HACK: chain together these disjoint sets. Workaround for the
                // fact that Rust's existential types don't play nice in
                // separate if branches, so cannot do something like
                //
                // if rank == 1 || rank == 8 {
                //   promoting_moves()
                // } else {
                //   single_move()
                // }
                promoting
                    .squares()
                    .flat_map(move |to| promoting_moves(from, to, color))
                    .chain(
                        not_promoting
                            .squares()
                            .map(move |to| Move::Single { from, to }),
                    )
            })
    }

    fn en_passant_captures<'a>(&'a self, color: Color) -> impl Iterator<Item = Move> + 'a {
        self.en_passant_target.into_iter().flat_map(move |to| {
            let en_passant_target = Bitboard::empty().set(to);

            // Compute the squares from which en passant capture could originate
            let en_passant_from = match color {
                WHITE => {
                    ((en_passant_target >> 7) & !A_FILE) | ((en_passant_target >> 9) & !H_FILE)
                }
                BLACK => {
                    ((en_passant_target << 7) & !H_FILE) | ((en_passant_target << 9) & !A_FILE)
                }
            };

            (en_passant_from & self.presence_for(color).pawn)
                .squares()
                .map(move |from| Move::Single { from, to })
        })
    }

    // All pawn threats. Note, these are not necessarily
    // legal moves: there must be an opponent on the attacked square.
    fn all_pawn_attacks<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Square, Bitboard)> + 'a {
        self.presence_for(color)
            .pawn
            .squares()
            .map(move |from| (from, self._pawn_attacks(from, &Piece(PAWN, color))))
    }

    fn _pawn_attacks(&self, from: Square, pawn: &Piece) -> Bitboard {
        let bitboard = bitboard![from];

        match pawn.color() {
            WHITE => {
                let east_attacks = bitboard.north_by(1).east_by(1) & !A_FILE;
                let west_attacks = bitboard.north_by(1).west_by(1) & !H_FILE;
                east_attacks | west_attacks
            }
            BLACK => {
                let east_attacks = bitboard.south_by(1).east_by(1) & !A_FILE;
                let west_attacks = bitboard.south_by(1).west_by(1) & !H_FILE;
                east_attacks | west_attacks
            }
        }
    }

    fn all_pawn_advances<'a>(&'a self, color: Color) -> impl Iterator<Item = Move> + 'a {
        fn promoting_moves(
            from: Square,
            target: Square,
            color: Color,
        ) -> impl Iterator<Item = Move> {
            [QUEEN, ROOK, BISHOP, KNIGHT]
                .into_iter()
                .map(move |new_piece| Move::Promote {
                    from,
                    to: target,
                    piece: Piece(new_piece, color),
                })
        }

        self.presence_for(color)
            .pawn
            .squares()
            .map(move |from| (from, self._pawn_advances(from, &Piece(PAWN, color))))
            .flat_map(move |(from, bitboard)| {
                // Only one of these is going to set anything
                let promoting = bitboard & (RANK_1 | RANK_8);
                let not_promoting = bitboard & !(RANK_1 | RANK_8);

                // Expand those attacked squares into concrete moves

                // HACK: chain together these disjoint sets. Workaround for the
                // fact that Rust's existential types don't play nice in
                // separate if branches, so cannot do something like
                //
                // if rank == 1 || rank == 8 {
                //   promoting_moves()
                // } else {
                //   single_move()
                // }
                promoting
                    .squares()
                    .flat_map(move |to| promoting_moves(from, to, color))
                    .chain(
                        not_promoting
                            .squares()
                            .map(move |to| Move::Single { from, to }),
                    )
            })
    }

    fn _pawn_advances(&self, from: Square, pawn: &Piece) -> Bitboard {
        let this_piece = bitboard![from];

        let start_rank = match pawn.color() {
            WHITE => Rank::_2,
            BLACK => Rank::_7,
        };

        let max_magnitude = if from.rank() == start_rank { 2 } else { 1 };

        let mut advances = Bitboard::empty();

        for magnitude in 1..=max_magnitude {
            let moved_forward = match pawn.color() {
                WHITE => this_piece.north_by(magnitude),
                BLACK => this_piece.south_by(magnitude),
            };

            let not_blocked = moved_forward & !(self.presence_black.all | self.presence_white.all);

            if not_blocked.is_empty() {
                break;
            } else {
                advances = advances | not_blocked
            }
        }

        advances
    }

    fn all_king_attacks<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Square, Bitboard)> + 'a {
        self.presence_for(color)
            .king
            .squares()
            .map(move |from| (from, self._king_attacks(from, &Piece(KING, color))))
    }

    // Note: does not exclude moves that put the king in check
    fn _king_attacks(&self, from: Square, king: &Piece) -> Bitboard {
        let attacks = PRECOMPUTED_BITBOARDS.king_moves[from.index()];

        let same_color = match king.color() {
            WHITE => self.presence_white.all,
            BLACK => self.presence_black.all,
        };

        attacks & !same_color
    }

    fn all_rook_attacks<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Square, Bitboard)> + 'a {
        self.presence_for(color)
            .rook
            .squares()
            .map(move |from| (from, self._rook_attacks(from, &Piece(ROOK, color))))
    }

    fn _rook_attacks(&self, from: Square, rook: &Piece) -> Bitboard {
        let same_color = match rook.color() {
            WHITE => self.presence_white.all,
            BLACK => self.presence_black.all,
        };

        let other_color = match rook.color() {
            WHITE => self.presence_black.all,
            BLACK => self.presence_white.all,
        };

        compute_attacks(
            from,
            &PRECOMPUTED_BITBOARDS.rays.north,
            same_color,
            other_color,
            |b| b.bitscan_forward_unchecked(),
            H8,
        ) | compute_attacks(
            from,
            &PRECOMPUTED_BITBOARDS.rays.east,
            same_color,
            other_color,
            |b| b.bitscan_forward_unchecked(),
            H8,
        ) | compute_attacks(
            from,
            &PRECOMPUTED_BITBOARDS.rays.south,
            same_color,
            other_color,
            |b| b.bitscan_backward_unchecked(),
            A1,
        ) | compute_attacks(
            from,
            &PRECOMPUTED_BITBOARDS.rays.west,
            same_color,
            other_color,
            |b| b.bitscan_backward_unchecked(),
            A1,
        )
    }

    fn all_bishop_attacks<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Square, Bitboard)> + 'a {
        self.presence_for(color)
            .bishop
            .squares()
            .map(move |from| (from, self._bishop_attacks(from, &Piece(BISHOP, color))))
    }

    fn _bishop_attacks(&self, from: Square, bishop: &Piece) -> Bitboard {
        let same_color = match bishop.color() {
            WHITE => self.presence_white.all,
            BLACK => self.presence_black.all,
        };

        let other_color = match bishop.color() {
            WHITE => self.presence_black.all,
            BLACK => self.presence_white.all,
        };

        compute_attacks(
            from,
            &PRECOMPUTED_BITBOARDS.rays.north_west,
            same_color,
            other_color,
            |b| b.bitscan_forward_unchecked(),
            H8,
        ) | compute_attacks(
            from,
            &PRECOMPUTED_BITBOARDS.rays.north_east,
            same_color,
            other_color,
            |b| b.bitscan_forward_unchecked(),
            H8,
        ) | compute_attacks(
            from,
            &PRECOMPUTED_BITBOARDS.rays.south_east,
            same_color,
            other_color,
            |b| b.bitscan_backward_unchecked(),
            A1,
        ) | compute_attacks(
            from,
            &PRECOMPUTED_BITBOARDS.rays.south_west,
            same_color,
            other_color,
            |b| b.bitscan_backward_unchecked(),
            A1,
        )
    }

    fn all_knight_attacks<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Square, Bitboard)> + 'a {
        self.presence_for(color)
            .knight
            .squares()
            .map(move |from| (from, self._knight_attacks(from, &Piece(KNIGHT, color))))
    }

    fn _knight_attacks<'a>(&'a self, from: Square, knight: &'a Piece) -> Bitboard {
        let attacks = PRECOMPUTED_BITBOARDS.knight_moves[from.index()];

        let same_color = match knight.color() {
            WHITE => self.presence_white.all,
            BLACK => self.presence_black.all,
        };

        attacks & !same_color
    }

    fn all_queen_attacks<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Square, Bitboard)> + 'a {
        self.presence_for(color)
            .queen
            .squares()
            .map(move |from| (from, self._queen_attacks(from, &Piece(QUEEN, color))))
    }

    fn _queen_attacks(&self, from: Square, queen: &Piece) -> Bitboard {
        self._rook_attacks(from, queen) | self._bishop_attacks(from, queen)
    }

    // TODO: remove. Only kept for testing
    #[cfg(test)]
    pub fn find_next_move(&self, max_depth: u8) -> Result<Option<(Move, Score)>> {
        let (mv, score, _, _) = self.find_best_move(max_depth)?;
        Ok(mv.map(|m| (m, score)))
    }

    pub fn find_best_move(
        &self,
        max_depth: u8,
    ) -> Result<(Option<Move>, Score, Vec<(Move, Color)>, u64)> {
        let mut pv: PV = Vec::new();
        let mut history: History = Vec::new();

        let curr_depth = 0;
        let (mv, score, node_count) = self._find_next_move_parallel(
            curr_depth,
            max_depth,
            &mut pv,
            &mut history,
            Score::MIN,
            Score::MAX,
        )?;

        let full_pv = full_history(pv, self.color_to_move);

        info!(
            "Main line score={}, path={}, node_count={}",
            score,
            HistoryDisplay(&full_pv),
            node_count
        );

        Ok((mv, score, full_pv, node_count))
    }

    fn _find_next_move(
        &self,
        curr_depth: u8,
        max_depth: u8,
        // Principal variation result, used to return the best line found in this node.
        // Though slightly confusing, it is more efficient to pass a result
        // buffer here instead of returning a new buffer from this function.
        pv: &mut PV,
        // List of all moves made up to this point in the search tree (all ancestors)
        history: &mut History,
        mut alpha: Score,
        mut beta: Score,
    ) -> Result<(Option<Move>, Score, u64)> {
        // Find all pieces
        // Generate all valid moves for those pieces.
        // After each move, must not be in check - prune.
        // Make each move - evaluate the position.
        // Pick highest scoring move

        // Leaf node, we are done.
        // Unless our king is in check, then extend the search by one ply.

        if curr_depth == max_depth && self.is_in_check(self.color_to_move)? {
            return self._find_next_move(curr_depth, max_depth + 1, pv, history, alpha, beta);
        } else if curr_depth == max_depth {
            return Ok((None, evaluate_position(self, history)?, 1));
        }

        let mut best_move: Option<Move> = None;
        let mut best_score = match self.color_to_move {
            WHITE => Score::MIN,
            BLACK => Score::MAX,
        };
        let mut node_count: u64 = 1; // number of nodes visited by all

        // Remember current location in history
        let history_len = history.len();

        // Will hold the PV for each child move
        let mut child_pv: PV = Vec::new();

        for mv in self.candidate_moves(history) {
            let moved_board = self.make_move(mv)?;

            // Add this node to the history, truncating to ensure sibling nodes
            // from previous iterations are removed.
            history.truncate(history_len);
            history.push((mv, self.color_to_move));

            // Clear any sibling PV
            child_pv.clear();

            debug!(
                "{}: Evaluating move {} initial score={} α={} β={}",
                self.color_to_move,
                HistoryDisplay(history),
                best_score,
                alpha,
                beta
            );

            // Cannot move into check. This helps verify that the position
            // is not checkmate: if the recursive call below returns no
            // moves then the position is mate
            //
            // TODO: this is confusing. Cleaner way to handle this? Filter all
            // moves to only return legal moves?
            if moved_board.is_in_check(self.color_to_move)? {
                debug!(
                    "{}: Continue. In check after illegal move {}",
                    self.color_to_move,
                    HistoryDisplay(history)
                );

                continue;
            }

            // Evaluate board score at leaf nodes
            let (_, score, subtree_node_count) = moved_board._find_next_move(
                curr_depth + 1,
                max_depth,
                &mut child_pv,
                history,
                alpha,
                beta,
            )?;

            node_count += subtree_node_count;

            // Ensure that only *fully-searched* paths are returned. A pruned path could
            // possibly be worse than the fully searched path. The alpha-beta bound guarantees
            // that pruned subtrees are *equivalent to or worse than* the original search path.

            match self.color_to_move {
                WHITE => {
                    alpha = cmp::max(alpha, score);
                    if score > best_score
                    // shortest line is best, this picks the quickest forced mate sequence
                       || (score == best_score && child_pv.len() + 1 < pv.len())
                    {
                        best_score = score;
                        best_move = Some(mv);

                        // Write new PV into parent's PV list
                        pv.clear();
                        pv.push(mv);
                        pv.extend(child_pv.iter());
                    }
                }
                BLACK => {
                    beta = cmp::min(beta, score);
                    if score < best_score || (score == best_score && child_pv.len() + 1 < pv.len())
                    {
                        best_score = score;
                        best_move = Some(mv);

                        pv.clear();
                        pv.push(mv);
                        pv.extend(child_pv.iter());
                    }
                }
            }

            debug!(
                "{}: Evaluated move {} score={} α={} β={} nodes={}",
                self.color_to_move,
                HistoryDisplay(history),
                score,
                alpha,
                beta,
                subtree_node_count
            );

            if alpha >= beta {
                debug!(
                    "{}: {} Found α={} >= β={}. Pruning rest of node.",
                    self.color_to_move,
                    HistoryDisplay(&history),
                    alpha,
                    beta
                );
                return Ok((Some(mv), score, node_count));
            }
        }

        // No legal moves, this position is either checkmate or stalemate
        if best_move.is_none() {
            if self.is_in_check(self.color_to_move)? {
                debug!("Position is checkmate for {}", self.color_to_move);
                return match self.color_to_move {
                    WHITE => Ok((None, Score::checkmate_white(), 1)),
                    BLACK => Ok((None, Score::checkmate_black(), 1)),
                };
            } else {
                debug!("Position is stalemate");
                return Ok((None, Score::ZERO, 1));
            }
        }

        Ok((best_move, best_score, node_count))
    }

    fn _find_next_move_parallel(
        &self,
        curr_depth: u8,
        max_depth: u8,
        // Principal variation result, used to return the best line found in this node.
        // Though slightly confusing, it is more efficient to pass a result
        // buffer here instead of returning a new buffer from this function.
        pv: &mut PV,
        // List of all moves made up to this point in the search tree (all ancestors)
        history: &mut History,
        alpha: Score,
        beta: Score,
    ) -> Result<(Option<Move>, Score, u64)> {
        if curr_depth == max_depth && self.is_in_check(self.color_to_move)? {
            return self._find_next_move(curr_depth, max_depth + 1, pv, history, alpha, beta);
        } else if curr_depth == max_depth {
            return Ok((None, evaluate_position(self, history)?, 1));
        }

        #[derive(Debug)]
        struct State {
            best_move: Option<Move>,
            best_score: Score,
            pv: PV,
            node_count: u64, // todo - atomic?
            alpha: Score,
            beta: Score,
        };

        let state_mutex = Mutex::new(State {
            best_move: None,
            best_score: match self.color_to_move {
                WHITE => Score::MIN,
                BLACK => Score::MAX,
            },
            pv: Vec::new(),
            node_count: 1,
            alpha: alpha,
            beta: beta,
        });

        let result = self
            .candidate_moves(history)
            .collect::<Vec<Move>>()
            .into_par_iter()
            .try_for_each(|mv| {
                let moved_board = self.make_move(mv)?;

                // Will hold the PV for each child move
                let mut child_pv: PV = Vec::new();
                let mut history: History = history.clone();
                history.push((mv, self.color_to_move));

                let state = state_mutex
                    .lock()
                    .map_err(|_| IOError("Could not lock mutex".to_string()))?;
                let alpha = state.alpha;
                let beta = state.beta;

                debug!(
                    "{}: Evaluating move {} initial score={} α={} β={}",
                    self.color_to_move,
                    HistoryDisplay(&history),
                    state.best_score,
                    alpha,
                    beta
                );

                drop(state); // be sure to drop mutex before long computation

                if moved_board.is_in_check(self.color_to_move)? {
                    debug!(
                        "{}: Continue. In check after illegal move {}",
                        self.color_to_move,
                        HistoryDisplay(&history)
                    );
                    return Ok(());
                }

                let (_, score, subtree_node_count) = if history.len() < 3 {
                    moved_board._find_next_move_parallel(
                        curr_depth + 1,
                        max_depth,
                        &mut child_pv,
                        &mut history,
                        alpha,
                        beta,
                    )?
                } else {
                    moved_board._find_next_move(
                        curr_depth + 1,
                        max_depth,
                        &mut child_pv,
                        &mut history,
                        alpha,
                        beta,
                    )?
                };

                let mut state = state_mutex
                    .lock()
                    .map_err(|_| IOError("Could not lock mutex".to_string()))?;

                state.node_count += subtree_node_count;

                match self.color_to_move {
                    WHITE => {
                        state.alpha = cmp::max(state.alpha, score);
                        if score > state.best_score
                            || (score == state.best_score && child_pv.len() + 1 < state.pv.len())
                        {
                            state.best_score = score;
                            state.best_move = Some(mv);

                            state.pv.clear();
                            state.pv.push(mv);
                            state.pv.extend(child_pv.iter());
                        }
                    }
                    BLACK => {
                        state.beta = cmp::min(state.beta, score);
                        if score < state.best_score
                            || (score == state.best_score && child_pv.len() + 1 < state.pv.len())
                        {
                            state.best_score = score;
                            state.best_move = Some(mv);

                            state.pv.clear();
                            state.pv.push(mv);
                            state.pv.extend(child_pv.iter());
                        }
                    }
                }

                debug!(
                    "{}: Evaluated move {} score={} α={} β={} nodes={}",
                    self.color_to_move,
                    HistoryDisplay(&history),
                    score,
                    state.alpha,
                    state.beta,
                    subtree_node_count
                );

                // TODO: how to shortcircuit?
                if state.alpha >= state.beta {
                    debug!(
                        "{}: {} Found α={} >= β={}. Pruning rest of node.",
                        self.color_to_move,
                        HistoryDisplay(&history),
                        state.alpha,
                        state.beta
                    );
                    return Err(Break);
                }

                Ok(())
            });

        match result {
            Ok(()) => (),
            Err(Break) => (),
            Err(e) => return Err(e),
        }

        let state = state_mutex
            .lock()
            .map_err(|_| IOError("Could not lock mutex".to_string()))?;

        if state.best_move.is_none() {
            if self.is_in_check(self.color_to_move)? {
                debug!("Position is checkmate for {}", self.color_to_move);
                return match self.color_to_move {
                    WHITE => Ok((None, Score::checkmate_white(), 1)),
                    BLACK => Ok((None, Score::checkmate_black(), 1)),
                };
            } else {
                debug!("Position is stalemate");
                return Ok((None, Score::ZERO, 1));
            }
        }

        // TODO: do this unconditionally?
        pv.clear();
        pv.extend(state.pv.iter());

        Ok((state.best_move, state.best_score, state.node_count))
    }

    // Open files are files containing no pawns

    pub fn open_files(&self) -> Bitboard {
        let pawns = self.presence_white.pawn | self.presence_black.pawn;
        let mut open = Bitboard::empty();

        for file in ALL_FILES {
            if (pawns & file).is_empty() {
                open |= file;
            }
        }

        open
    }

    pub fn checkmate(&self) -> Result<bool> {
        let color = self.color_to_move;

        if !self.is_in_check(color)? {
            return Ok(false);
        }

        debug!("{}: In check. Evaluating for checkmate", color);

        for mv in self.all_moves() {
            let moved_board = self.make_move(mv)?;

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
                let piece: Option<Piece> = self.piece_on_square(Square::new(*file, *rank));

                let symbol = match piece {
                    Some(piece) => square_symbol(&piece),
                    None => " ".to_string(),
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

pub fn square_symbol(p: &Piece) -> String {
    let Piece(piece, color) = *p;

    let uncolored = piece.to_string();

    match color {
        BLACK => uncolored,
        WHITE => uncolored.to_ascii_uppercase(),
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

#[cfg(test)]
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
        .place_piece(Piece(BISHOP, WHITE), F1);

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
        .all_moves()
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
        .all_moves()
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
        .all_moves()
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
        .all_moves()
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
        .all_moves()
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
        .all_moves()
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
        .all_moves()
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
        .all_moves()
        .collect::<Vec<Move>>()
        .contains(&Move::CastleKingside));

    let castled_board = board.make_move(Move::CastleKingside).unwrap();

    println!("{}", board.presence_black.king);
    println!("{}", castled_board.presence_black.king);
    println!("{}", castled_board.presence_black.all);

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
        .all_moves()
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

    assert!(!moved_board.can_castle_kingside(WHITE));
    assert!(moved_board.can_castle_kingside(BLACK));
    assert!(!moved_board.can_castle_queenside(WHITE));
    assert!(moved_board.can_castle_queenside(BLACK));
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

    assert!(moved_board.can_castle_kingside(WHITE));
    assert!(!moved_board.can_castle_kingside(BLACK));
    assert!(moved_board.can_castle_queenside(WHITE));
    assert!(!moved_board.can_castle_queenside(BLACK));
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

    assert!(moved_board.can_castle_kingside(WHITE));
    assert!(moved_board.can_castle_kingside(BLACK));
    assert!(!moved_board.can_castle_queenside(WHITE));
    assert!(moved_board.can_castle_queenside(BLACK));
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

    assert!(moved_board.can_castle_kingside(WHITE));
    assert!(moved_board.can_castle_kingside(BLACK));
    assert!(moved_board.can_castle_queenside(WHITE));
    assert!(!moved_board.can_castle_queenside(BLACK));
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

    assert!(moved_board.can_castle_kingside(WHITE));
    assert!(!moved_board.can_castle_kingside(BLACK));
    assert!(moved_board.can_castle_queenside(WHITE));
    assert!(moved_board.can_castle_queenside(BLACK));
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

    assert!(!moved_board.can_castle_kingside(WHITE));
    assert!(moved_board.can_castle_kingside(BLACK));
    assert!(moved_board.can_castle_queenside(WHITE));
    assert!(moved_board.can_castle_queenside(BLACK));
}

#[test]
fn test_should_castle() {
    init();
    let board =
        crate::fen::parse("r2qkbnr/ppp2ppp/2np4/4p3/4P1b1/5N2/PPPPBPPP/RNBQK2R w KQkq - 2 5")
            .unwrap();

    let (_, _, path, _) = board.find_best_move(6).unwrap();

    assert!(path
        .iter()
        .find(|(mv, color)| (*mv == Move::CastleKingside && *color == WHITE))
        .is_some());
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

    assert!(!board.is_empty(A1));
    assert!(!board.is_empty(C6));
    assert!(board.is_empty(D3));
}

#[test]
fn test_checkmate() {
    let board = Board::empty()
        .with_color_to_move(BLACK)
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(ROOK, WHITE), B2)
        .place_piece(Piece(KING, BLACK), A6);

    assert!(board.checkmate().unwrap());
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
    assert!(!board.checkmate().unwrap());
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
fn test_checkmate_opponent_king_and_rook_foo() {
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

    println!("{}", board1);

    let (mv, _) = board1.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv, Move::new(B7, H7));

    let board2 = board1.make_move(mv).unwrap();

    // Only move available
    let board3 = board2.make_move((H8, G8).into()).unwrap();

    let (mv3, _) = board3.find_next_move(3).unwrap().unwrap();
    assert_eq!(mv3, Move::new(A7, G7));

    let board4 = board3.make_move(mv3).unwrap();
    assert!(board4.checkmate().unwrap());

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

    println!("{}", board1);

    let (mv, _) = board1.find_next_move(4).unwrap().unwrap();
    assert_eq!(mv, Move::new(B7, H7));

    let board2 = board1.make_move(mv).unwrap();

    // Only move available
    let board3 = board2.make_move((H8, G8).into()).unwrap();

    let (mv3, _) = board3.find_next_move(4).unwrap().unwrap();
    assert_eq!(mv3, Move::new(A7, G7));

    let board4 = board3.make_move(mv3).unwrap();
    assert!(board4.checkmate().unwrap());

    println!("{}", board1);
    println!("{}", board2);
    println!("{}", board3);
    println!("{}", board4);
}

#[test]
fn test_avoid_stalemate() {
    init();
    let board = crate::fen::parse("6k1/6p1/4K1P1/2p4r/2P3P1/2Pq4/8/8 b - - 3 62").unwrap();

    let (mv, _) = board.find_next_move(4).unwrap().unwrap();
    // Would be stalemate:
    assert_ne!(mv, Move::Single { from: D3, to: D8 });

    Puzzle::new(board)
        .should_find_move(H5, G5)
        .respond_with(E6, E7)
        .should_find_move(G5, E5)
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
        assert!(self.board.checkmate().unwrap());
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

//     let board = crate::fen::parse("q1r4k/6pp/8/3Q2N1/8/8/6PP/7K w - - 0 1").unwrap();

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

    println!("{}", board.presence_white.pawn);
    println!("{}", board.presence_white.all);
    println!("{}", moved_board.presence_white.pawn);
    println!("{}", moved_board.presence_white.queen);
    println!("{}", moved_board.presence_white.all);

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

    println!("{}", board);
    dbg!(board.legal_moves(G5).unwrap());
    assert!(board
        .legal_moves(G5)
        .unwrap()
        .contains(&Move::Single { from: G5, to: F6 }));

    assert!(board
        .legal_moves(E5)
        .unwrap()
        .contains(&Move::Single { from: E5, to: F6 }));

    let captured_board = board.make_move(Move::Single { from: E5, to: F6 }).unwrap();

    assert!(captured_board.is_empty(E5)); // capturing pawn has moved
    assert!(captured_board.is_empty(F5)); // captured pawn is gone
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

    assert!(captured_board.is_empty(A4)); // capturing pawn has moved
    assert!(captured_board.is_empty(B4)); // captured pawn is gone
    assert!(captured_board.piece_on_square(B3) == Some(Piece(PAWN, BLACK))); // capturer moved to en passant target
}

#[test]
fn test_en_passant_do_not_wrap_board() {
    let board =
        crate::fen::parse("rnbqkbnr/1ppppppp/8/p7/7P/8/PPPPPPP1/RNBQKBNR w KQkq a6 0 2").unwrap();

    assert_eq!(
        sorted(board.legal_moves(H4).unwrap()),
        sorted(vec![(H4, H5).into()])
    );
}

#[test]
fn test_castle_bug() {
    let board =
        crate::fen::parse("r3k2r/p1ppqNb1/1n2pnp1/1b1P4/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 1 2")
            .unwrap();

    // capture kingside rook with knight
    let moved_board = board.make_move((F7, H8).into()).unwrap();

    assert!(!moved_board.can_castle_kingside(BLACK));
    assert!(moved_board
        .all_moves()
        .find(|mv| *mv == Move::CastleKingside)
        .is_none());
}

#[test]
fn test_foo() {
    let board = crate::fen::parse("5rk1/3n2pp/2p1p3/5pP1/1PPP4/8/5PP1/R5K1 b - - 0 26").unwrap();

    board.find_next_move(4).unwrap();
}

#[test]
fn test_capture_and_promote_must_update_castle_rights() {
    // Bug: was producing the line G2H1b 0-0 H1F3 and crashing

    let board =
        crate::fen::parse("r3k2r/p1pN1pb1/bn3np1/8/1p2P3/2N2Q2/PPPBBPpP/R3K2R b KQkq - 0 3")
            .unwrap();

    board.find_next_move(4).unwrap();
}

#[test]
fn compute_open_files() {
    let board =
        crate::fen::parse("3q1rk1/1pp1p1pp/2np4/8/6b1/2N2N2/1PPPP1P1/R1BQK2R w KQ - 0 1").unwrap();
    assert_eq!(board.open_files(), A_FILE | F_FILE)
}
