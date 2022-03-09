use crate::board::*;
use crate::color::Color;
use crate::error::BoardError;
use crate::fen;
use crate::mv::*;
use crate::square::*;
use log::info;
use std::fmt;
use std::io::{self, Write};

use nom::{
    self,
    branch::alt,
    bytes::complete::tag,
    character::complete::{one_of, space1},
    combinator::{map, opt},
    multi::many1,
    sequence::{preceded, tuple},
    Finish, IResult,
};

/*
 * Main event loop for UCI interface.
 *
 * UCI protocol: http://page.mi.fu-berlin.de/block/uci.htm
 */
pub fn run_uci() -> Result<()> {
    info!("UCI: starting new session");

    let mut interface = UCIInterface::new();

    loop {
        let mut buffer = String::new();
        read_line(&mut buffer)?;

        let command = parse_uci_input(&buffer)?;
        let responses = interface.run_command(command)?;

        for response in responses.into_iter().flatten() {
            respond(response)?;
        }
    }
}

// UCI state machine
struct UCIInterface {
    board: Option<Board>,
}

impl UCIInterface {
    fn new() -> UCIInterface {
        UCIInterface { board: None }
    }

    fn run_command(&mut self, command: UCICommand) -> Result<Option<Vec<UCIResponse>>> {
        match command {
            UCICommand::UCI => Ok(Some(vec![
                UCIResponse::Id,
                // TODO: options
                UCIResponse::Option("name Hash type spin default 1 min 1 max 128".to_string()),
                UCIResponse::Ok,
            ])),
            UCICommand::SetOption => Ok(None), // TODO
            UCICommand::IsReady => Ok(Some(vec![UCIResponse::ReadyOk])),
            UCICommand::Position(mut new_board, moves) => {
                if let Some(moves) = moves {
                    for mv in moves.iter() {
                        // This is a little hacky. UCI specifies castling moves
                        // as e1g1 (for kingside white). In order to convert that
                        // to a `Move` variant, we need to introspect the board
                        // here. TODO: use the UCI format internally?

                        let fixed_move = match mv {
                            // white
                            Move::Single { from, to }
                                if *from == E1
                                    && *to == G1
                                    && new_board.piece_on_square(E1)
                                        == Some(Piece(KING, Color::WHITE)) =>
                            {
                                Move::CastleKingside
                            }
                            Move::Single { from, to }
                                if *from == E1
                                    && *to == C1
                                    && new_board.piece_on_square(E1)
                                        == Some(Piece(KING, Color::WHITE)) =>
                            {
                                Move::CastleQueenside
                            }
                            // black
                            Move::Single { from, to }
                                if *from == E8
                                    && *to == G8
                                    && new_board.piece_on_square(E8)
                                        == Some(Piece(KING, Color::BLACK)) =>
                            {
                                Move::CastleKingside
                            }
                            Move::Single { from, to }
                                if *from == E8
                                    && *to == C8
                                    && new_board.piece_on_square(E8)
                                        == Some(Piece(KING, Color::BLACK)) =>
                            {
                                Move::CastleQueenside
                            }

                            // Same hackiness with promotion. We don't know the
                            // color of the piece when parsing a promotion,
                            // update it here.
                            Move::Promote {
                                from,
                                to,
                                piece: Piece(unit, _),
                            } => Move::Promote {
                                from: *from,
                                to: *to,
                                piece: Piece(*unit, new_board.color_to_move),
                            },
                            _ => *mv,
                        };

                        new_board = new_board.make_move(fixed_move)?;
                    }
                }

                self.board = Some(new_board); // set state

                Ok(None)
            }
            UCICommand::Go => {
                if let Some(board) = self.board {
                    let depth = 4; //  TODO configure depth, with options?
                    let mv = board.find_next_move(depth)?;

                    if let Some((mv, _)) = mv {
                        Ok(Some(vec![UCIResponse::BestMove { mv: mv }]))
                    } else {
                        Err(BoardError::ProtocolError(
                            "No move found. In checkmate?".to_string(),
                        ))
                    }
                } else {
                    Err(BoardError::ProtocolError("No board setup".to_string()))
                }
            }
            UCICommand::Quit => Err(BoardError::NotImplemented),
            UCICommand::Stop => Ok(None), // TODO
        }
    }
}

fn read_line(buffer: &mut String) -> Result<()> {
    io::stdin()
        .read_line(buffer)
        .map_err(|e| BoardError::IOError(format!("IO error: {}", e)))?;
    info!("UCI: got input {}", buffer);
    Ok(())
}

// TODO why does this panic?
// position startpos moves e2e4 a7a6 d2d4 a6a5 d4d5 e7e5 g1f3 a5a4 b1c3 c7c6 c1e3 a4a3 f1c4 a3b2 a1b1 d8a5 e3d2 a5c5 d1e2 c5a5 b1b2 a5c5 d2e3 c5a5 e3d2 a5c5 f3e5 c5d6 e1g1 d6e5 d5c6

fn parse_uci_input(input: &str) -> Result<UCICommand> {
    let uci = map(tag("uci"), |_: &str| UCICommand::UCI);

    let is_ready = map(tag("isready"), |_: &str| UCICommand::IsReady);

    let set_option = map(tag("setoption"), |_: &str| UCICommand::SetOption);

    let quit = map(tag("quit"), |_: &str| UCICommand::Quit);

    let stop = map(tag("stop"), |_: &str| UCICommand::Stop);

    let go = map(tag("go"), |_: &str| UCICommand::Go);

    fn position(i: &str) -> IResult<&str, UCICommand> {
        let (i, _) = tag("position")(i)?;
        let (i, _) = space1(i)?;

        fn fen(i: &str) -> IResult<&str, Board> {
            let (i, _) = tag("fen")(i)?;
            let (i, _) = space1(i)?;
            fen::fen_parser(i)
        }

        fn startpos(i: &str) -> IResult<&str, Board> {
            let (i, _) = tag("startpos")(i)?;
            // TODO: better way to initialize starting board?
            let board =
                fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap(); // TODO no unwrap
            Ok((i, board))
        }

        let (i, board) = alt((fen, startpos))(i)?;

        // Finally, parse optional `moves`, for example:
        // > position startpos moves e2e4 a7a6 d2d4

        // TODO: ensure promotion can be parsed
        fn mv(i: &str) -> IResult<&str, Move> {
            let (i, (from, to, promotion)) = tuple((
                fen::square_parser,
                fen::square_parser,
                opt(map(one_of("nbrq"), |c| match c {
                    'n' => KNIGHT,
                    'b' => BISHOP,
                    'r' => ROOK,
                    'q' => QUEEN,
                    _ => panic!(""),
                })),
            ))(i)?;

            Ok((
                i,
                match promotion {
                    Some(piece) => Move::Promote {
                        from,
                        to,
                        // TODO: HACK we don't know the proper color of the piece here
                        piece: Piece(piece, crate::color::WHITE),
                    },
                    None => Move::Single { from, to },
                },
            ))
        }

        let (i, maybe_moves) = opt(preceded(
            preceded(space1, tag("moves")),
            many1(preceded(space1, mv)),
        ))(i)?;

        Ok((i, UCICommand::Position(board, maybe_moves)))
    }

    alt((uci, set_option, quit, stop, is_ready, position, go))(input)
        .finish()
        .map(|(_, command)| {
            info!("UCI: got command {:?}", command);
            command
        })
        .map_err(|e: nom::error::Error<&str>| {
            BoardError::ParseError(format!("Could not parse UCI input: {}", e))
        })
}

#[derive(Debug)]
enum UCICommand {
    UCI,
    Position(Board, Option<Vec<Move>>),
    SetOption, // TODO: parse name / value
    Quit,
    IsReady,
    Stop,
    Go, // TODO: parse options
}

#[derive(Debug)]
enum UCIResponse {
    Id,
    Ok,
    ReadyOk,
    BestMove { mv: Move },
    Option(String), // TODO actually implement
}

// serialize Engine -> GUI response
impl fmt::Display for UCIResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UCIResponse::Id => write!(f, "id name chess-engine id author Bo"),
            UCIResponse::Ok => write!(f, "uciok"),
            UCIResponse::ReadyOk => write!(f, "readyok"),
            UCIResponse::BestMove { mv } => write!(f, "bestmove {}", mv.to_string().to_lowercase()), // UCI really wants lowercase moves
            UCIResponse::Option(s) => write!(f, "option {}", s),
        }?;

        write!(f, "\n")
    }
}

fn respond(msg: UCIResponse) -> Result<()> {
    info!("Response: {}", msg);
    io::stdout()
        .write_all(msg.to_string().as_bytes())
        .map_err(|e| BoardError::IOError(format!("IO error: {}", e)))
}

#[cfg(test)]
fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_king_cannot_be_captured_by_pawn_promotion() {
    init();
    let mut interface = UCIInterface::new();

    // rnb1kbnr/1p1P1ppp/8/5q2/2B1P3/2N5/PRPBQPPP/5RK1 b kq - 0 17
    let commands = vec![
        "position startpos moves e2e4 a7a6 d2d4 a6a5 d4d5 e7e5 g1f3 a5a4 b1c3 c7c6 c1e3 a4a3 f1c4 a3b2 a1b1 d8a5 e3d2 a5c5 d1e2 c5a5 b1b2 a5c5 d2e3 c5a5 e3d2 a5c5 f3e5 c5d6 e1g1 d6e5 d5c6 e5f5 c6d7",
        "go"
    ];

    for command_str in commands.iter() {
        let command = parse_uci_input(command_str).unwrap();
        interface.run_command(command).unwrap();
    }
}
