use std::fmt;

/*
 * TODO: 0- or 1-indexing?
 */

const EMPTY: u8 = 0b00000000;
const PAWN: u8   = 0b00000001;
const KNIGHT: u8 = 0b00000010;   
const BISHOP: u8 = 0b00000011;
const ROOK: u8 = 0b000000100;
const QUEEN: u8 = 0b00000101;
const KING: u8 = 0b00000110;

const PIECE_MASK: u8 = 0b00000111;


const BLACK: u8 = 0b00000000;
const WHITE: u8 = 0b10000000;

const COLOR_MASK: u8 = 0b10000000;

#[derive(Debug)]
enum Piece {
    A = 0b00000001,
    B = 0b00000010,
}

#[derive(Clone)]
struct Board {
    board: [u8; 64],
}

impl Board {

    fn move_piece(self, from: Square, to: Square) -> Board {
	let mut new = self.clone();

	let i = (from.row * 8 + from.col) as usize;
	let j = (to.row * 8  + from.col) as usize;
	
	new.board[j] = new.board[i];  // TODO: check there is a piece here
	new.board[i] = 0;
	new
    }
}

fn square_symbol(p: u8) -> char {
    let uncolored = match p & PIECE_MASK {
	EMPTY => ' ',
	PAWN => 'p',
	KNIGHT => 'n',
	BISHOP => 'b',
	ROOK => 'r',
	QUEEN => 'q',
	KING => 'k',
        unknown => panic!("Unknown piece {}", unknown)
    };

    match p & COLOR_MASK {
	BLACK => uncolored,
	WHITE => uncolored.to_ascii_uppercase(),
	unknown => panic!("Unknown color {}, not possible", unknown)
    }
}

// TODO: bound checking
struct Square {
    row: u8,
    col: u8
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in (0..8).rev() {
//            write!(f, "{}│ ", row + 1)?;
            write!(f, "{} ", row + 1)?;	    
	    
            for col in 0..8 {
                let piece = self.board[row * 8 + col];
                write!(f, "{} ", square_symbol(piece))?;
            }
            write!(f, "\n")?;
        }

        // write!(f, "  ")?;	
	// for _ in 0..(8*2) {
        //     write!(f, "─")?;
	// }
        // write!(f, "\n")?;	
	
        for c in vec![' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] {
            write!(f, "{} ", c)?;
        }

        fmt::Result::Ok(())
    }
}

fn main() {
    println!("Hello, world!");

    let mut board: [u8; 64] = [0; 64];
//    println!("{:?}", board);

    let mut b = Board { board: board };
//    println!("{}", b);

    b.board[1] = PAWN | WHITE;
    b.board[2] = KNIGHT | WHITE;
    b.board[3] = BISHOP | WHITE;
    b.board[50] = BISHOP | BLACK;
    b.board[51] = PAWN | BLACK;            

    println!("{}", b);
    println!("{}", b.move_piece(Square{row:0, col:2}, Square{row:1, col:2}));


    println!("{}", PAWN);
    println!("{}", KNIGHT);
    println!("{}", BISHOP);
    println!("{}", ROOK);                
    println!("{}", QUEEN);            
    println!("{}", KING);

    println!("{:?}", Piece::A);
    println!("{}", square_symbol(QUEEN));
	
}
