use crate::board::*;
use crate::color::*;
use crate::error::BoardError;
use crate::file::*;
use crate::rank::*;
use crate::square::*;
use log::{debug, error};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete,
    character::complete::{digit1, one_of, space1},
    combinator::{map, map_res},
    multi::{count, many1},
    sequence::{pair, terminated},
    Finish, IResult,
};

#[derive(Debug, Clone)]
enum ParseResult {
    Piece(Piece),
    Spaces(usize),
}

impl ParseResult {
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
 * Parse FEN notation into a Board representation.
 */
pub fn parse(fen: &str) -> Result<Board> {
    fen_parser(fen).finish().map(|(_, b)| b).map_err(|e| {
        error!("{:?}", e);
        BoardError::ParseError(format!("Could not parse FEN '{}': {}", fen, e))
    })
}

/*
 * Internal FEN parser implementation using Nom. Can be composed with other
 * parsers.
 *
 * https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
 * https://ia802908.us.archive.org/26/items/pgn-standard-1994-03-12/PGN_standard_1994-03-12.txt
 */
pub fn fen_parser(fen: &str) -> IResult<&str, Board> {
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
        alt((map(square_parser, Some), map(complete::char('-'), |_| None))),
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

fn square_parser(input: &str) -> IResult<&str, Square> {
    map(
        pair(one_of("abcdefghABCDEFGH"), one_of("12345678")),
        |(f, r)| {
            Square::new(
                File::from_str(&*f.to_string()),
                Rank::from_str(&*r.to_string()),
            )
        },
    )(input)
}

pub fn parse_square(input: &str) -> Result<Square> {
    square_parser(input)
        .finish()
        .map(|(_, b)| b)
        .map_err(|e| BoardError::ParseError(format!("Could not parse square '{}': {}", input, e)))
}

#[cfg(test)]
fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_parse_start_position() {
    init();

    let fen = String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let board = parse(&fen).unwrap();

    let expected = Board::empty()
        .with_color_to_move(WHITE)
        .with_fullmove_clock(1)
        .with_halfmove_clock(0)
        .with_en_passant_target(None)
        .with_castle_rights(CastleRights::all())
        .place_piece(Piece(ROOK, BLACK), A8)
        .place_piece(Piece(KNIGHT, BLACK), B8)
        .place_piece(Piece(BISHOP, BLACK), C8)
        .place_piece(Piece(QUEEN, BLACK), D8)
        .place_piece(Piece(KING, BLACK), E8)
        .place_piece(Piece(BISHOP, BLACK), F8)
        .place_piece(Piece(KNIGHT, BLACK), G8)
        .place_piece(Piece(ROOK, BLACK), H8)
        .place_piece(Piece(PAWN, BLACK), A7)
        .place_piece(Piece(PAWN, BLACK), B7)
        .place_piece(Piece(PAWN, BLACK), C7)
        .place_piece(Piece(PAWN, BLACK), D7)
        .place_piece(Piece(PAWN, BLACK), E7)
        .place_piece(Piece(PAWN, BLACK), F7)
        .place_piece(Piece(PAWN, BLACK), G7)
        .place_piece(Piece(PAWN, BLACK), H7)
        .place_piece(Piece(PAWN, WHITE), A2)
        .place_piece(Piece(PAWN, WHITE), B2)
        .place_piece(Piece(PAWN, WHITE), C2)
        .place_piece(Piece(PAWN, WHITE), D2)
        .place_piece(Piece(PAWN, WHITE), E2)
        .place_piece(Piece(PAWN, WHITE), F2)
        .place_piece(Piece(PAWN, WHITE), G2)
        .place_piece(Piece(PAWN, WHITE), H2)
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(KNIGHT, WHITE), B1)
        .place_piece(Piece(BISHOP, WHITE), C1)
        .place_piece(Piece(QUEEN, WHITE), D1)
        .place_piece(Piece(KING, WHITE), E1)
        .place_piece(Piece(BISHOP, WHITE), F1)
        .place_piece(Piece(KNIGHT, WHITE), G1)
        .place_piece(Piece(ROOK, WHITE), H1);

    assert_eq!(board, expected);
}

#[test]
fn test_parse_start_e4() {
    init();

    let fen = String::from("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let board = parse(&fen).unwrap();

    let expected = Board::empty()
        .with_color_to_move(BLACK)
        .with_fullmove_clock(1)
        .with_halfmove_clock(0)
        .with_en_passant_target(Some(E3))
        .with_castle_rights(CastleRights::all())
        .place_piece(Piece(ROOK, BLACK), A8)
        .place_piece(Piece(KNIGHT, BLACK), B8)
        .place_piece(Piece(BISHOP, BLACK), C8)
        .place_piece(Piece(QUEEN, BLACK), D8)
        .place_piece(Piece(KING, BLACK), E8)
        .place_piece(Piece(BISHOP, BLACK), F8)
        .place_piece(Piece(KNIGHT, BLACK), G8)
        .place_piece(Piece(ROOK, BLACK), H8)
        .place_piece(Piece(PAWN, BLACK), A7)
        .place_piece(Piece(PAWN, BLACK), B7)
        .place_piece(Piece(PAWN, BLACK), C7)
        .place_piece(Piece(PAWN, BLACK), D7)
        .place_piece(Piece(PAWN, BLACK), E7)
        .place_piece(Piece(PAWN, BLACK), F7)
        .place_piece(Piece(PAWN, BLACK), G7)
        .place_piece(Piece(PAWN, BLACK), H7)
        .place_piece(Piece(PAWN, WHITE), A2)
        .place_piece(Piece(PAWN, WHITE), B2)
        .place_piece(Piece(PAWN, WHITE), C2)
        .place_piece(Piece(PAWN, WHITE), D2)
        .place_piece(Piece(PAWN, WHITE), E4)
        .place_piece(Piece(PAWN, WHITE), F2)
        .place_piece(Piece(PAWN, WHITE), G2)
        .place_piece(Piece(PAWN, WHITE), H2)
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(KNIGHT, WHITE), B1)
        .place_piece(Piece(BISHOP, WHITE), C1)
        .place_piece(Piece(QUEEN, WHITE), D1)
        .place_piece(Piece(KING, WHITE), E1)
        .place_piece(Piece(BISHOP, WHITE), F1)
        .place_piece(Piece(KNIGHT, WHITE), G1)
        .place_piece(Piece(ROOK, WHITE), H1);

    assert_eq!(board, expected);
}

#[test]
#[allow(non_snake_case)]
fn test_parse_start_e4_c5_Nf3() {
    init();

    let fen = String::from("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
    let board = parse(&fen).unwrap();

    let expected = Board::empty()
        .with_color_to_move(BLACK)
        .with_fullmove_clock(2)
        .with_halfmove_clock(1)
        .with_en_passant_target(None)
        .with_castle_rights(CastleRights::all())
        .place_piece(Piece(ROOK, BLACK), A8)
        .place_piece(Piece(KNIGHT, BLACK), B8)
        .place_piece(Piece(BISHOP, BLACK), C8)
        .place_piece(Piece(QUEEN, BLACK), D8)
        .place_piece(Piece(KING, BLACK), E8)
        .place_piece(Piece(BISHOP, BLACK), F8)
        .place_piece(Piece(KNIGHT, BLACK), G8)
        .place_piece(Piece(ROOK, BLACK), H8)
        .place_piece(Piece(PAWN, BLACK), A7)
        .place_piece(Piece(PAWN, BLACK), B7)
        .place_piece(Piece(PAWN, BLACK), C5)
        .place_piece(Piece(PAWN, BLACK), D7)
        .place_piece(Piece(PAWN, BLACK), E7)
        .place_piece(Piece(PAWN, BLACK), F7)
        .place_piece(Piece(PAWN, BLACK), G7)
        .place_piece(Piece(PAWN, BLACK), H7)
        .place_piece(Piece(PAWN, WHITE), A2)
        .place_piece(Piece(PAWN, WHITE), B2)
        .place_piece(Piece(PAWN, WHITE), C2)
        .place_piece(Piece(PAWN, WHITE), D2)
        .place_piece(Piece(PAWN, WHITE), E4)
        .place_piece(Piece(PAWN, WHITE), F2)
        .place_piece(Piece(PAWN, WHITE), G2)
        .place_piece(Piece(PAWN, WHITE), H2)
        .place_piece(Piece(ROOK, WHITE), A1)
        .place_piece(Piece(KNIGHT, WHITE), B1)
        .place_piece(Piece(BISHOP, WHITE), C1)
        .place_piece(Piece(QUEEN, WHITE), D1)
        .place_piece(Piece(KING, WHITE), E1)
        .place_piece(Piece(BISHOP, WHITE), F1)
        .place_piece(Piece(KNIGHT, WHITE), F3)
        .place_piece(Piece(ROOK, WHITE), H1);

    assert_eq!(board, expected);
}
