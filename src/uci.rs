use crate::board::*;
use crate::error::BoardError;
use crate::fen::fen_parser;
use log::info;
use std::fmt;
use std::io::{self, Write};

use nom::{
    self, branch::alt, bytes::complete::tag, character::complete::space1, combinator::map, Finish,
    IResult,
};

/*
 * Main event loop for UCI interface.
 */
pub fn run_uci() -> Result<()> {
    info!("UCI: starting new session");

    let mut board: Option<Board> = None;

    while true {
        let mut buffer = String::new();
        read_line(&mut buffer)?;
        let command = parse_uci_input(&buffer)?;

        match command {
            UCICommand::UCI => {
                respond(UCIResponse::Id)?;
                // TODO: options
                respond(UCIResponse::Ok)
            }
            UCICommand::SetOption => Ok(()), // TODO
            UCICommand::IsReady => respond(UCIResponse::ReadyOk),
            UCICommand::Position(new_board) => {
                board = Some(new_board); // set state
                Ok(())
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

    fn position(i: &str) -> IResult<&str, UCICommand> {
        let (i, _) = tag("position")(i)?;
        let (i, _) = space1(i)?;
        // TODO: handle startpos
        let (i, _) = tag("fen")(i)?;
        let (i, _) = space1(i)?;

        let (i, board) = fen_parser(i)?;

        Ok((i, UCICommand::Position(board)))
    }

    alt((uci, set_option, quit, stop, is_ready, position))(input)
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
}

#[derive(Debug)]
enum UCIResponse {
    Id,
    Ok,
    ReadyOk,
}

// serialize Engine -> GUI response
impl fmt::Display for UCIResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UCIResponse::Id => write!(f, "id name chess-engine id author Bo"),
            UCIResponse::Ok => write!(f, "uciok"),
            UCIResponse::ReadyOk => write!(f, "readyok"),
        }
    }
}

fn respond(msg: UCIResponse) -> Result<()> {
    info!("Response: {}", msg);
    io::stdout()
        .write_all(msg.to_string().as_bytes())
        .map_err(|e| BoardError::IOError(format!("IO error: {}", e)))
}
