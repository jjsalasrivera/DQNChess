use std::io;

use chess::{Piece, Color, Square, Board, Rank, File};
use crossterm::{execute, terminal::{Clear, ClearType}, cursor::MoveTo};
use tch::Tensor;

pub fn display_board(board: &Board) {
    match board.side_to_move() {
        Color::Black => display_board_black_side(board),
        Color::White => display_board_white_side(board),
    }
}

fn display_board_black_side(board: &Board) {
    _ = execute!(io::stdout(), Clear(ClearType::All), MoveTo(0, 0));
    let mut fila = 0;
    for rank in 0..=7 {
        print!(" {} ", rank + 1);
        for file in (0..=7).rev() {
            let square = Square::make_square(Rank::from_index(rank), File::from_index(file));
            let piece = board.piece_on(square);
            let color = board.color_on(square);

            let symbol = match piece {
                Some(p) => match color {
                    Some(Color::White) => match p {
                        Piece::Pawn => "♟",
                        Piece::Knight => "♞",
                        Piece::Bishop => "♝",
                        Piece::Rook => "♜",
                        Piece::Queen => "♛",
                        Piece::King => "♚",
                    },
                    Some(Color::Black) => match p {
                        Piece::Pawn => "♙",
                        Piece::Knight => "♘",
                        Piece::Bishop => "♗",
                        Piece::Rook => "♖",
                        Piece::Queen => "♕",
                        Piece::King => "♔",
                    },
                    None => "",
                },
                None => "·",
            };

            print!(" {} ", symbol);
        }
        fila += 1;
        _ = execute!(io::stdout(), MoveTo(0, fila));
    }
    _ = execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveTo(0, fila));
    println!("    h  g  f  e  d  c  b  a");
    _ = execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveTo(0, fila+1));
}

fn display_board_white_side(board: &Board) {
    _ = execute!(io::stdout(), Clear(ClearType::All), MoveTo(0, 0));
    let mut fila = 0;
    for rank in (0..=7).rev() {
        print!(" {} ", rank + 1);
        for file in 0..=7 {
            let square = Square::make_square(Rank::from_index(rank), File::from_index(file));
            let piece = board.piece_on(square);
            let color = board.color_on(square);

            let symbol = match piece {
                Some(p) => match color {
                    Some(Color::Black) => match p {
                        Piece::Pawn => "♙",
                        Piece::Knight => "♘",
                        Piece::Bishop => "♗",
                        Piece::Rook => "♖",
                        Piece::Queen => "♕",
                        Piece::King => "♔",
                    },
                    Some(Color::White) => match p {
                        Piece::Pawn => "♟",
                        Piece::Knight => "♞",
                        Piece::Bishop => "♝",
                        Piece::Rook => "♜",
                        Piece::Queen => "♛",
                        Piece::King => "♚",
                    },
                    None => "",
                },
                None => "·",
            };

            print!(" {} ", symbol);
        }
        fila += 1;
        _ = execute!(io::stdout(), MoveTo(0, fila));
    }
    _ = execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveTo(0, fila));
    println!("    a  b  c  d  e  f  g  h");
    _ = execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveTo(0, fila+1));
}

pub fn board_to_tensor(board: &Board) -> Tensor {
    let mut state = [[0; 8]; 9];

    for rank in 0..=7 {
        for file in 0..=7 {
            let square = Square::make_square(Rank::from_index(rank), File::from_index(file));
            let piece_opt = board.piece_on(square);

            if let Some(piece) = piece_opt {
                let pv = piece.to_index() as i32 + 1;

                state[rank][file] = match board.color_on(square).unwrap() {
                    Color::Black => -pv,
                    Color::White => pv,
                };
            }
        }
    }

    state[8][0] = match board.side_to_move() {
        Color::Black => -1,
        Color::White => 1,
    };

    let my_castle_rights = match board.my_castle_rights() {
        chess::CastleRights::NoRights => 0,
        chess::CastleRights::KingSide => 1,
        chess::CastleRights::QueenSide => 2,
        chess::CastleRights::Both => 3,
    };

    let their_castle_rights = match board.their_castle_rights() {
        chess::CastleRights::NoRights => 0,
        chess::CastleRights::KingSide => 1,
        chess::CastleRights::QueenSide => 2,
        chess::CastleRights::Both => 3,
    };

    if board.side_to_move() == Color::White {
        state[8][1] = my_castle_rights;
        state[8][2] = their_castle_rights;
    }
    else {
        state[8][2] = my_castle_rights;
        state[8][1] = their_castle_rights;
    }

    let vector: Vec<i32> = state.iter().flat_map(|f| f.iter()).cloned().collect();
    Tensor::from_slice(vector.as_slice())
}