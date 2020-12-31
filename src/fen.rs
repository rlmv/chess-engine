use crate::board::*;
use crate::color::*;
use crate::square::*;
use log::debug;
use std::convert::TryFrom;

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete,
    character::complete::{digit1, one_of, space1},
    combinator::{map, map_res},
    multi::{count, many1},
    sequence::{pair, terminated},
    IResult,
};

#[derive(Debug, Clone)]
enum ParseResult {
    Piece(Piece),
    Spaces(usize),
}

impl ParseResult {
    // TODO - can something in NOM do this?
    fn spaces(c: char) -> std::result::Result<ParseResult, std::num::ParseIntError> {
        String::from(c).parse().map(ParseResult::Spaces)
    }

    fn piece(c: char) -> ParseResult {
        let color = if c.is_ascii_uppercase() { WHITE } else { BLACK };

        let mut lower = c.clone();
        lower.make_ascii_lowercase();

        let piece = match lower {
            'p' => PAWN,
            'n' => KNIGHT,
            'b' => BISHOP,
            'r' => ROOK,
            'q' => QUEEN,
            'k' => KING,
            _ => panic!(),
        };

        ParseResult::Piece(Piece(piece, color))
    }
}
/*
 * Parse FEN notation.
 *
 * https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
 * https://ia802908.us.archive.org/26/items/pgn-standard-1994-03-12/PGN_standard_1994-03-12.txt
 */
pub fn parse(fen: &str) -> IResult<&str, Board> {
    debug!("Parsing FEN: {}", fen);

    let pieces = map(one_of("rnbkqpRNBKQP"), ParseResult::piece);
    let spaces = map_res(one_of("12345678"), ParseResult::spaces);

    let line = many1(alt((pieces, spaces)));
    let line_end = alt((tag("/"), space1));
    let terminated_line = terminated(line, line_end);

    // TODO: separated_list instead?
    let (input, ranks) = count(terminated_line, 8)(fen)?;

    // TODO: assert that all 64 squares are represented

    let board = ranks
        .iter()
        .rev()
        .flat_map(|rank| rank.iter())
        // expand spaces into individual elements
        .flat_map(|tag| match tag {
            ParseResult::Spaces(x) => vec![None; *x],
            ParseResult::Piece(p) => vec![Some(p.clone())],
        })
        // ordered and indexed A1,A2,etc
        .enumerate()
        // populate the board
        .fold(Board::empty(), |board, (i, piece)| match piece {
            Some(p) => board.place_piece(p, Square::from_index(i)),
            _ => board,
        });

    // next to move

    let (input, color_to_move) = map(terminated(one_of("wb"), space1), |c| match c {
        'w' => WHITE,
        'b' => BLACK,
        _ => panic!(), // not possible
    })(input)?;

    debug!("Color to move: {:?}", color_to_move);

    // castling

    let (input, castling_rights) = terminated(
        alt((
            map(many1(one_of("kqKQ")), |cs| CastleRights {
                kingside_black: cs.iter().find(|&&c| c == 'k').is_some(),
                queenside_black: cs.iter().find(|&&c| c == 'q').is_some(),
                kingside_white: cs.iter().find(|&&c| c == 'K').is_some(),
                queenside_white: cs.iter().find(|&&c| c == 'Q').is_some(),
            }),
            map(complete::char('-'), |_| CastleRights::none()),
        )),
        space1,
    )(input)?;

    debug!("{:?}", castling_rights);

    // en passant

    let (input, en_passant_target) = terminated(
        alt((
            map_res(pair(one_of("abcdefgh"), one_of("12345678")), |(f, r)| {
                let s = format!("{}{}", f, r);
                Square::try_from(&*s).map(Some)
            }),
            map(complete::char('-'), |_| None),
        )),
        space1,
    )(input)?;

    debug!("En passant target: {:?}", en_passant_target);

    // move clocks
    let (input, halfmove_clock) =
        map_res(terminated(digit1, space1), |s: &str| s.parse::<u16>())(input)?;

    debug!("Halfmove: {:?}", halfmove_clock);

    let (input, fullmove_clock) = map_res(digit1, |s: &str| s.parse::<u16>())(input)?;

    debug!("Fullmove: {:?}", fullmove_clock);

    Ok((
        input,
        board
            .with_color_to_move(color_to_move)
            .with_en_passant_target(en_passant_target)
            .with_halfmove_clock(halfmove_clock)
            .with_fullmove_clock(fullmove_clock)
            .with_castle_rights(castling_rights),
    ))
}

#[cfg(test)]
fn square(s: &str) -> Square {
    Square::try_from(s).unwrap()
}

#[cfg(test)]
fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_parse_start_position() {
    init();

    let fen = String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let (_, board) = parse(&fen).unwrap();

    let expected = Board::empty()
        .with_color_to_move(WHITE)
        .with_fullmove_clock(1)
        .with_halfmove_clock(0)
        .with_en_passant_target(None)
        .with_castle_rights(CastleRights::all())
        .place_piece(Piece(ROOK, BLACK), square("A8"))
        .place_piece(Piece(KNIGHT, BLACK), square("B8"))
        .place_piece(Piece(BISHOP, BLACK), square("C8"))
        .place_piece(Piece(QUEEN, BLACK), square("D8"))
        .place_piece(Piece(KING, BLACK), square("E8"))
        .place_piece(Piece(BISHOP, BLACK), square("F8"))
        .place_piece(Piece(KNIGHT, BLACK), square("G8"))
        .place_piece(Piece(ROOK, BLACK), square("H8"))
        .place_piece(Piece(PAWN, BLACK), square("A7"))
        .place_piece(Piece(PAWN, BLACK), square("B7"))
        .place_piece(Piece(PAWN, BLACK), square("C7"))
        .place_piece(Piece(PAWN, BLACK), square("D7"))
        .place_piece(Piece(PAWN, BLACK), square("E7"))
        .place_piece(Piece(PAWN, BLACK), square("F7"))
        .place_piece(Piece(PAWN, BLACK), square("G7"))
        .place_piece(Piece(PAWN, BLACK), square("H7"))
        .place_piece(Piece(PAWN, WHITE), square("A2"))
        .place_piece(Piece(PAWN, WHITE), square("B2"))
        .place_piece(Piece(PAWN, WHITE), square("C2"))
        .place_piece(Piece(PAWN, WHITE), square("D2"))
        .place_piece(Piece(PAWN, WHITE), square("E2"))
        .place_piece(Piece(PAWN, WHITE), square("F2"))
        .place_piece(Piece(PAWN, WHITE), square("G2"))
        .place_piece(Piece(PAWN, WHITE), square("H2"))
        .place_piece(Piece(ROOK, WHITE), square("A1"))
        .place_piece(Piece(KNIGHT, WHITE), square("B1"))
        .place_piece(Piece(BISHOP, WHITE), square("C1"))
        .place_piece(Piece(QUEEN, WHITE), square("D1"))
        .place_piece(Piece(KING, WHITE), square("E1"))
        .place_piece(Piece(BISHOP, WHITE), square("F1"))
        .place_piece(Piece(KNIGHT, WHITE), square("G1"))
        .place_piece(Piece(ROOK, WHITE), square("H1"));

    assert_eq!(board, expected);
}

#[test]
fn test_parse_start_e4_c5_Nf3() {
    init();

    let fen = String::from("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
    let (_, board) = parse(&fen).unwrap();

    let expected = Board::empty()
        .with_color_to_move(BLACK)
        .with_fullmove_clock(2)
        .with_halfmove_clock(1)
        .with_en_passant_target(None)
        .with_castle_rights(CastleRights::all())
        .place_piece(Piece(ROOK, BLACK), square("A8"))
        .place_piece(Piece(KNIGHT, BLACK), square("B8"))
        .place_piece(Piece(BISHOP, BLACK), square("C8"))
        .place_piece(Piece(QUEEN, BLACK), square("D8"))
        .place_piece(Piece(KING, BLACK), square("E8"))
        .place_piece(Piece(BISHOP, BLACK), square("F8"))
        .place_piece(Piece(KNIGHT, BLACK), square("G8"))
        .place_piece(Piece(ROOK, BLACK), square("H8"))
        .place_piece(Piece(PAWN, BLACK), square("A7"))
        .place_piece(Piece(PAWN, BLACK), square("B7"))
        .place_piece(Piece(PAWN, BLACK), square("C5"))
        .place_piece(Piece(PAWN, BLACK), square("D7"))
        .place_piece(Piece(PAWN, BLACK), square("E7"))
        .place_piece(Piece(PAWN, BLACK), square("F7"))
        .place_piece(Piece(PAWN, BLACK), square("G7"))
        .place_piece(Piece(PAWN, BLACK), square("H7"))
        .place_piece(Piece(PAWN, WHITE), square("A2"))
        .place_piece(Piece(PAWN, WHITE), square("B2"))
        .place_piece(Piece(PAWN, WHITE), square("C2"))
        .place_piece(Piece(PAWN, WHITE), square("D2"))
        .place_piece(Piece(PAWN, WHITE), square("E4"))
        .place_piece(Piece(PAWN, WHITE), square("F2"))
        .place_piece(Piece(PAWN, WHITE), square("G2"))
        .place_piece(Piece(PAWN, WHITE), square("H2"))
        .place_piece(Piece(ROOK, WHITE), square("A1"))
        .place_piece(Piece(KNIGHT, WHITE), square("B1"))
        .place_piece(Piece(BISHOP, WHITE), square("C1"))
        .place_piece(Piece(QUEEN, WHITE), square("D1"))
        .place_piece(Piece(KING, WHITE), square("E1"))
        .place_piece(Piece(BISHOP, WHITE), square("F1"))
        .place_piece(Piece(KNIGHT, WHITE), square("F3"))
        .place_piece(Piece(ROOK, WHITE), square("H1"));

    assert_eq!(board, expected);
}
