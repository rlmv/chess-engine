use crate::board::*;
use crate::error::BoardError;
use crate::fen::fen_parser;
use crate::square::*;
use log::info;
use std::fmt;
use std::io::{self, Write};

use nom::{
    self, branch::alt, bytes::complete::tag, character::complete::space1, combinator::map, Finish,
    IResult,
};

/*
 * Main event loop for UCI interface.
 *
 * UCI protocol: http://page.mi.fu-berlin.de/block/uci.htm
 */
pub fn run_uci() -> Result<()> {
    info!("UCI: starting new session");

    let mut board: Option<Board> = None;

    loop {
        let mut buffer = String::new();
        read_line(&mut buffer)?;
        let command = parse_uci_input(&buffer)?;

        match command {
            UCICommand::UCI => {
                respond(UCIResponse::Id)?;
                // TODO: options
                respond(UCIResponse::Option(
                    "name Hash type spin default 1 min 1 max 128".to_string(),
                ))?;
                respond(UCIResponse::Ok)
            }
            UCICommand::SetOption => Ok(()), // TODO
            UCICommand::IsReady => respond(UCIResponse::ReadyOk),
            UCICommand::Position(new_board) => {
                board = Some(new_board); // set state
                Ok(())
            }
            UCICommand::Go => {
                if let Some(board) = board {
                    let depth = 4; //  TODO configure depth
                    let mv = board.find_next_move(depth)?;

                    if let Some((mv, _)) = mv {
                        respond(UCIResponse::BestMove { mv: mv })
                    } else {
                        Err(BoardError::ProtocolError(
                            "No move found. In checkmate?".to_string(),
                        ))
                    }
                } else {
                    Err(BoardError::ProtocolError("No board setup".to_string()))
                }
            }
            UCICommand::Quit => break,
            UCICommand::Stop => Ok(()), // TODO
            other => Err(BoardError::ProtocolError(format!(
                "Got unexpected command {:?}",
                other,
            ))),
        }?;
    }

    Ok(())
}

fn read_line(buffer: &mut String) -> Result<()> {
    io::stdin()
        .read_line(buffer)
        .map_err(|e| BoardError::IOError(format!("IO error: {}", e)))?;
    info!("UCI: got input {}", buffer);
    Ok(())
}

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
        // TODO: handle startpos
        let (i, _) = tag("fen")(i)?;
        let (i, _) = space1(i)?;

        let (i, board) = fen_parser(i)?;

        Ok((i, UCICommand::Position(board)))
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
    Position(Board),
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
            UCIResponse::BestMove { mv } => write!(f, "bestmove {}", mv),
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
