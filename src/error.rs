use std::fmt;

use crate::square::Square;

use BoardError::*;

#[derive(Debug)]
pub enum BoardError {
    NoPieceOnFromSquare(Square),
    NotImplemented,
    IllegalState(String),
    ParseError(String),
    IOError(String),
    ProtocolError(String),
    IllegalCastle, // TODO provide reason
}

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoPieceOnFromSquare(square) => write!(f, "Square {:?} does not have a piece", square),
            NotImplemented => write!(f, "Missing implementation"),
            IllegalState(msg) => write!(f, "{}", msg),
            ParseError(msg) => write!(f, "{}", msg),
            IOError(msg) => write!(f, "{}", msg),
            ProtocolError(msg) => write!(f, "{}", msg),
            IllegalCastle => write!(f, "Not allowed to castle in this position"),
        }
    }
}

// impl From<regex::Error> for BoardError {
//     // TODO: chain errors
//     fn from(re: regex::Error) -> Self {
//         BoardError::ParseError(format!("{:?}", re))
//     }
// }
