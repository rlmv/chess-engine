use crate::board::{Board, Result};
use crate::color::*;
use crate::fen;
use crate::mv::Move;
#[cfg(test)]
use crate::square::*;
#[cfg(test)]
use colored::Colorize;
#[cfg(test)]
use itertools::Itertools;
use paste;
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

// Perft function
//
// Results here: https://www.chessprogramming.org/Perft_Results

#[allow(dead_code)]
fn perft(board: Board, depth: usize) -> Result<usize> {
    if depth == 0 {
        return Ok(1);
    }

    let mut count = 0;

    for mv in board.all_moves(board.color_to_move)? {
        let moved_board = board.make_move(mv)?;

        // Drop illegal moves
        if moved_board.is_in_check(board.color_to_move)? {
            continue;
        }

        count += perft(moved_board, depth - 1)?;
    }

    return Ok(count);
}

#[allow(dead_code)]
fn perft_debug(board: Board, depth: usize) -> Result<HashMap<Move, usize>> {
    assert!(depth >= 1);

    let mut moves: HashMap<Move, usize> = HashMap::new();

    for mv in board.all_moves(board.color_to_move)? {
        let moved_board = board.make_move(mv)?;

        // Drop illegal moves
        if moved_board.is_in_check(board.color_to_move)? {
            continue;
        }

        moves.insert(mv, perft(moved_board, depth - 1)?);
    }

    return Ok(moves);
}

#[cfg(test)]
fn stockfish_perft(fen: &str, moves: &Vec<Move>, depth: usize) -> HashMap<String, usize> {
    let mut child = Command::new("stockfish")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn child process");

    let mut stdin = child.stdin.take().expect("failed to get stdin");

    let mut input = format!("position fen {}", fen);
    if moves.len() > 0 {
        input += " moves";
        for mv in moves.iter() {
            input += " ";
            input += &mv.to_string().to_lowercase();
        }
    }
    input += "\n";

    stdin
        .write_all(input.as_bytes())
        .expect("Failed to write to stdin");

    stdin
        .write_all(format!("go perft {}\n", depth).as_bytes())
        .expect("Failed to write to stdin");

    stdin
        .write_all("quit\n".as_bytes())
        .expect("Failed to write to stdin");

    let output = child.wait_with_output().expect("Failed to read stdout");
    let response = String::from_utf8_lossy(&output.stdout);

    assert!(response.len() > 0);

    let moves: HashMap<String, usize> = response
        .split("\n")
        .filter(|line| {
            !(line.is_empty()
                || line.starts_with("Stockfish")
                || line.starts_with("Nodes searched"))
        })
        // line looks like "b1c3: 440"
        .map(|line| {
            let no_colon = line.replace(":", " ");
            let mut parts = no_colon.split_whitespace().map(|part| part.to_string());

            let mv = parts.next().expect("Did not find move");
            let count = parts
                .next()
                .expect("Did not find move")
                .parse()
                .expect("Failed to parse node count");

            (mv, count)
        })
        .collect();

    moves
}

#[cfg(test)]
fn compare_to_stockfish(fen: &str, setup_moves: Vec<Move>, depth: usize) -> () {
    let stockfish_moves: HashMap<String, usize> = stockfish_perft(fen, &setup_moves, depth);

    let mut board = fen::parse(fen).expect("Could not parse FEN string");
    for mv in setup_moves.into_iter() {
        board = board.make_move(mv).unwrap();
    }
    let color = board.color_to_move;

    let our_moves: HashMap<String, usize> = perft_debug(board, depth)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (format_move(k, color), v))
        .collect();

    // union together moves

    let mut combined_moves: Vec<(String, (Option<&usize>, Option<&usize>))> = stockfish_moves
        .keys()
        .chain(our_moves.keys())
        .unique()
        .map(|mv| (mv.clone(), (stockfish_moves.get(mv), our_moves.get(mv))))
        .collect();

    combined_moves.sort_by_key(|(mv, _)| mv.clone());

    println!("Move    Stockfish     chess-engine");
    println!("----    ---------     ------------");

    for (mv, (stockfish_count, count)) in combined_moves.into_iter() {
        let mut line = format!(
            "{:<8}{:>5}{:>15}",
            mv,
            format_count(stockfish_count),
            format_count(count)
        );
        if stockfish_count != count || stockfish_count.is_none() || count.is_none() {
            line = line.red().to_string()
        }

        println!("{}", line);
    }

    assert!(stockfish_moves == our_moves);
}

#[cfg(test)]
fn format_count(count: Option<&usize>) -> String {
    count.map_or("∅".to_string(), |c| c.to_string())
}

// format a move to match Stockfish's format
#[cfg(test)]
fn format_move(mv: Move, color: Color) -> String {
    let s = match mv {
        Move::Single { from: _, to: _ } => mv.to_string(),
        Move::Promote {
            from: _,
            to: _,
            piece: _,
        } => mv.to_string(),
        Move::CastleKingside if color == WHITE => "e1g1".to_string(),
        Move::CastleKingside => "e8g8".to_string(),
        Move::CastleQueenside if color == WHITE => "e1c1".to_string(),
        Move::CastleQueenside => "e8c8".to_string(),
    };

    s.to_lowercase()
}

/*
 * Macro to generate a perft test for a given position and depth
 */
#[macro_export]
macro_rules! perft_test {
    ( fen=$fen:ident, depth=$depth:literal ) => {
        paste::item! {
            #[test]
            fn [<$fen:lower _depth_ $depth>]() {
                compare_to_stockfish($fen, Vec::new(), $depth);
            }
        }
    };
}

#[cfg(test)]
const POSITION_1: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[cfg(test)]
const POSITION_2: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0";

#[cfg(test)]
const POSITION_4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";

perft_test!(fen = POSITION_1, depth = 1);
perft_test!(fen = POSITION_1, depth = 2);
perft_test!(fen = POSITION_1, depth = 3);
perft_test!(fen = POSITION_1, depth = 4);
// perft_test!(fen = POSITION_1, depth = 5);
// perft_test!(fen = POSITION_1, depth = 6);

perft_test!(fen = POSITION_2, depth = 1);
perft_test!(fen = POSITION_2, depth = 2);
perft_test!(fen = POSITION_2, depth = 3);
// perft_test!(fen = POSITION_2, depth = 4);

perft_test!(fen = POSITION_4, depth = 4);
