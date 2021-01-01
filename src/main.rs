use chess_engine::fen;

use std::env;

const DEFAULT_DEPTH: u8 = 3;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let mut args_iter = args.iter();

    //    println!("Arguments: {:?}", args);

    let _ = args_iter.next(); // Drop binary name

    let board = if let Some(fen) = args_iter.next() {
        fen::parse(fen).unwrap()
    } else {
        panic!("Expected a FEN string");
    };

    let depth: u8 = args_iter
        .next()
        .map(|s| s.parse().unwrap())
        .unwrap_or(DEFAULT_DEPTH);

    println!("Parsed board: \n\n{}\n", board);
    println!("Searching for best move to depth {}...\n", depth);

    let mv = board.find_next_move(depth).unwrap();

    match mv {
        None => println!("No move found. In checkmate?"),
        Some(mv) => {
            println!("Done. Best move is {}{}\n", mv.from, mv.to);
            let moved_board = board.move_piece(mv.from, mv.to).unwrap();
            println!("{}\n", moved_board);
        }
    }
}
