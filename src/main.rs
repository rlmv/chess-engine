use chess_engine::fen;
use chess_engine::uci;

use std::env;

use log::error;
use log::LevelFilter;
// use log4rs::append::console::{ConsoleAppender, Target};
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};

const DEFAULT_DEPTH: u8 = 3;

fn main() {
    configure_logging();

    let args: Vec<String> = env::args().collect();
    let mut args_iter = args.iter();

    //    println!("Arguments: {:?}", args);

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

fn configure_logging() -> () {
    //    let stderr = ConsoleAppender::builder().target(Target::Stderr).build();

    let file = FileAppender::builder()
        .build("/home/bo/.scidvspc/log/internal.log")
        .unwrap();

    let config = Config::builder()
        //        .appender(Appender::builder().build("stderr", Box::new(stderr)))
        .appender(Appender::builder().build("file", Box::new(file)))
        .build(
            Root::builder()
                //                .appender("stderr")
                .appender("file")
                .build(LevelFilter::Debug),
        )
        .unwrap();

    log4rs::init_config(config).unwrap();
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
