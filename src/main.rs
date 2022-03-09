use chess_engine::error::BoardError;
use chess_engine::fen;
use chess_engine::uci;

use std::env;

use log::error;
use log::LevelFilter;
use log4rs::append::console::{ConsoleAppender, Target};
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log_panics;

const DEFAULT_DEPTH: u8 = 4;

fn main() {
    match configure_logging() {
        Ok(_) => (),
        Err(e) => panic!("{}", e),
    };

    let args: Vec<String> = env::args().collect();
    let mut args_iter = args.iter();

    let _ = args_iter.next(); // Drop binary name

    let command = args_iter
        .next()
        .map(|command| command.to_lowercase())
        .map(|command| {
            if command == "fen" {
                Command::FEN
            } else if command == "uci" {
                Command::UCI
            } else {
                panic!("Did not recognize CLI command {}", command)
            }
        });

    match command {
        Some(Command::FEN) => {
            if let Some(fen) = args_iter.next() {
                let depth: u8 = args_iter
                    .next()
                    .map(|s| s.parse().unwrap())
                    .unwrap_or(DEFAULT_DEPTH);
                evaluate_fen(fen, depth)
            } else {
                panic!("Expected a FEN string");
            }
        }
        Some(Command::UCI) | None => uci::run_uci().map_err(|e| error!("{}", e)).unwrap(),
    };
}

enum Command {
    UCI,
    FEN,
}

fn configure_logging() -> Result<(), BoardError> {
    log_panics::init();

    let log_level = match env::var("LOG_LEVEL") {
        Ok(l) if l == "DEBUG" => LevelFilter::Debug,
        Ok(l) if l == "INFO" => LevelFilter::Info,
        Ok(l) => Err(BoardError::ConfigError(format!("Unknown LOG_LEVEL: {}", l)))?,
        Err(_) => LevelFilter::Info,
    };

    let log_stdout = env::var("LOG_STDOUT").is_ok();

    let file = FileAppender::builder()
        .build("/home/bo/Code/chess-engine/out.log")
        .map_err(|e| BoardError::ConfigError(e.to_string()))?;

    let stderr = ConsoleAppender::builder().target(Target::Stderr).build();

    let mut config_builder =
        Config::builder().appender(Appender::builder().build("file", Box::new(file)));
    if log_stdout {
        config_builder =
            config_builder.appender(Appender::builder().build("stderr", Box::new(stderr)));
    }

    let mut root_builder = Root::builder().appender("file");
    if log_stdout {
        root_builder = root_builder.appender("stderr");
    }

    let config = config_builder
        .build(root_builder.build(log_level))
        .map_err(|e| BoardError::ConfigError(e.to_string()))?;

    log4rs::init_config(config).map_err(|e| BoardError::ConfigError(e.to_string()))?;

    Ok(())
}

fn evaluate_fen(fen: &String, depth: u8) -> () {
    let board = fen::parse(fen).unwrap();

    println!("Parsed board: \n\n{}\n", board);
    println!("Searching for best move to depth {}...\n", depth);

    let mv = board.find_next_move(depth).unwrap();

    match mv {
        None => println!("No move found. In checkmate?"),
        Some((mv, _)) => {
            println!("Done. Best move is {}\n", mv);
            let moved_board = board.make_move(mv).unwrap();
            println!("{}\n", moved_board);
        }
    }
}
