use chess_engine::board::*;
use chess_engine::color::*;
use chess_engine::file::*;
use chess_engine::rank::*;
use chess_engine::square::*;

fn main() {
    env_logger::init();

    let b = Board::empty()
        .place_piece(Piece(PAWN, WHITE), Square::new(File::A, Rank::_2))
        .place_piece(Piece(KNIGHT, WHITE), Square::new(File::B, Rank::_1))
        .place_piece(Piece(BISHOP, WHITE), Square::new(File::C, Rank::_1))
        .place_piece(Piece(KING, WHITE), Square::new(File::E, Rank::_1))
        .place_piece(Piece(BISHOP, BLACK), Square::new(File::C, Rank::_8))
        .place_piece(Piece(KING, BLACK), Square::new(File::E, Rank::_8));

    println!("{}", b);
    println!(
        "{}",
        b.move_piece(
            Square::new(File::A, Rank::_2),
            Square::new(File::A, Rank::_4)
        )
        .unwrap()
    );

    let b2 = Board::empty()
        .place_piece(Piece(ROOK, WHITE), Square::new(File::A, Rank::_2))
        .place_piece(Piece(KING, WHITE), Square::new(File::E, Rank::_1))
        .place_piece(Piece(ROOK, BLACK), Square::new(File::C, Rank::_8))
        .place_piece(Piece(KING, BLACK), Square::new(File::E, Rank::_8));

    println!("{}", b2);
    println!("{:?}", b2.find_next_move(1).unwrap());
}
