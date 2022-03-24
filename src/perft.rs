use crate::board::{Board, Result};
use crate::fen;
use crate::mv::Move;
use crate::square::*;
#[cfg(test)]
use colored::Colorize;
#[cfg(test)]
use itertools::Itertools;
use std::collections::HashMap;

use std::io::Write;
use std::process::{Command, Stdio};

// Perft function
//
// Results here: https://www.chessprogramming.org/Perft_Results

fn perft(board: Board, depth: usize) -> Result<usize> {
    if depth == 0 {
        return Ok(1);
    }

    let mut count = 0;

    for mv in board.all_moves(board.color_to_move)? {
        let moved_board = board.make_move(mv)?;

        // Drop illegal moves
        if moved_board.is_in_check(board.color_to_move)? {
            dbg!(mv);
            continue;
        }

        count += perft(moved_board, depth - 1)?;
    }

    return Ok(count);
}

fn perft_debug(board: Board, depth: usize) -> Result<HashMap<Move, usize>> {
    assert!(depth >= 1);

    let mut moves: HashMap<Move, usize> = HashMap::new();

    for mv in board.all_moves(board.color_to_move)? {
        let moved_board = board.make_move(mv)?;

        // Drop illegal moves
        if moved_board.is_in_check(board.color_to_move)? {
            dbg!(mv);
            continue;
        }

        moves.insert(mv, perft(moved_board, depth - 1)?);
    }

    return Ok(moves);
}

#[cfg(test)]
fn stockfish_perft(moves: &Vec<Move>, depth: usize) -> HashMap<String, usize> {
    let mut child = Command::new("stockfish")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn child process");

    let mut stdin = child.stdin.take().expect("failed to get stdin");

    let mut input = "position startpos".to_string();
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
fn startpos() -> Board {
    fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
}

#[cfg(test)]
fn compare_to_stockfish(setup_moves: Vec<Move>, depth: usize) -> () {
    let stockfish_moves: HashMap<String, usize> = stockfish_perft(&setup_moves, depth);

    let mut board = startpos();
    for mv in setup_moves.into_iter() {
        board = board.make_move(mv).unwrap();
    }

    let our_moves: HashMap<String, usize> = perft_debug(board, depth)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k.to_string().to_lowercase(), v))
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

#[test]
fn perft_1() {
    assert_eq!(perft(startpos(), 1).unwrap(), 20);
}

#[test]
fn perft_2() {
    assert_eq!(perft(startpos(), 2).unwrap(), 400)
}

#[test]
fn perft_3() {
    let setup_moves = vec![
        Move::Single { from: H2, to: H4 },
        Move::Single { from: A7, to: A5 },
    ];
    let depth = 1;

    compare_to_stockfish(setup_moves, depth);

    assert_eq!(perft(startpos(), 3).unwrap(), 8902)
}

#[test]
fn perft_4() {
    assert_eq!(perft(startpos(), 4).unwrap(), 197281)
}
