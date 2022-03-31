use chess_engine::error::BoardError;
use chess_engine::fen;
use chess_engine::perft;
use chess_engine::uci;
use chess_engine::util::localize;
use log::error;
use log::LevelFilter;
use log4rs::append::console::{ConsoleAppender, Target};
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log_panics;
use std::env;
use std::time::Instant;

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
            } else if command == "perft" {
                Command::PERFT
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
        Some(Command::PERFT) => {
            if let Some(fen) = args_iter.next() {
                let depth: u8 = args_iter
                    .next()
                    .map(|s| s.parse().unwrap())
                    .unwrap_or(DEFAULT_DEPTH);
                run_perft(fen, depth)
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
    PERFT,
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

    let start = Instant::now();

    let result = board.find_best_move(depth).unwrap();

    let elapsed = start.elapsed();

    match result {
        (None, _, _, _) => println!("No move found. In checkmate?"),
        (Some(mv), _, _, node_count) => {
            println!("Done. Best move is {}\n", mv);
            println!(
                "Evaluated {} nodes in {} seconds. {} nodes per second.\n",
                localize(node_count),
                elapsed.as_secs(),
                localize(node_count as u64 / elapsed.as_secs())
            );
            let moved_board = board.make_move(mv).unwrap();
            println!("{}\n", moved_board);
        }
    }
}

fn run_perft(fen: &String, depth: u8) -> () {
    let board = fen::parse(fen).unwrap();

    println!("Parsed board: \n\n{}\n", board);
    println!("Running perft to depth {}...\n", depth);

    let start = Instant::now();

    let result = perft::perft(board, depth.into()).unwrap();

    let elapsed = start.elapsed();

    println!(
        "Done. Evaluated {} nodes in {} seconds. {} nodes per second.",
        localize(result),
        elapsed.as_secs(),
        localize(result as u64 / elapsed.as_secs())
    );
}
