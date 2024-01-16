use chess::{Board, Color, File, MoveGen, Piece, Rank, Square, ChessMove, GameResult};

const PAWN_VALUE: i32 = 100;
const KNIGHT_VALUE: i32 = 320;
const BISHOP_VALUE: i32 = 350;
const ROOK_VALUE: i32 = 500;
const QUEEN_VALUE: i32 = 900;
const KING_VALUE: i32 = 5000;

pub const MAX_REWARD: f32 = 10000.0;

const KING_PROXIMITY_CENTER: [[i32; 8]; 8] = [
    [-10, -8, -6, -4, -4, -6, -8, -10],
    [-8, -4,  0,  2,  2,  0, -4, -8],
    [-6,  0,  4,  6,  6,  4,  0, -6],
    [-4,  2,  6,  8,  8,  6,  2, -4],
    [-4,  2,  6,  8,  8,  6,  2, -4],
    [-6,  0,  4,  6,  6,  4,  0, -6],
    [-8, -4,  0,  2,  2,  0, -4, -8],
    [-10, -8, -6, -4, -4, -6, -8, -10],
];

#[derive(PartialEq, Clone)]
struct PieceComplete {
    piece: Piece,
    color: Color,
    square_opt: Option<Square>,
}

pub fn evaluate(board: &Board, num_moves: u32, game_result: Option<GameResult>) -> f32 {
    /*
    - Situacion de mate
    - Piezas bando - piezas contrarias
    - Peones encadenados
    - Piezas amenazadas
    - Piezas defendidas
    - Apertura (10 primeros movimientos) Control de centro
    - Finales (num piezas <= 15) Rey en el centro
        */

    let res = match game_result {
        Some(GameResult::WhiteCheckmates | GameResult::BlackResigns) => MAX_REWARD,
        Some(GameResult::BlackCheckmates | GameResult::WhiteResigns) => -MAX_REWARD,
        Some(GameResult::DrawAccepted | GameResult::DrawDeclared | GameResult::Stalemate) => 0.0,
        None => {
            let pieces = get_pieces(board);
            let legal_moves = MoveGen::new_legal(board).collect::<Vec<ChessMove>>();

            let c_pieces = count_pieces(&pieces);
            let king_position = king_position(board, &pieces);
            let threats = count_threats(board, &pieces, &legal_moves);
            let available_squares = count_available_squares(board, &pieces);
            let opening = if num_moves <= 20 {
                opening(board, &pieces, &legal_moves)
            } 
            else {
                0
            };
            
            (c_pieces + opening + king_position + threats + available_squares) as f32
        }  
    };

    res / MAX_REWARD

}

fn opening(board: &Board, pieces: &[PieceComplete], moves: &[ChessMove]) -> i32 {
    let mut res = 0;

    let board_reversed_opt = board.null_move();
    let total_moves: &[ChessMove];
    let binding: Vec<ChessMove>;

    if let Some(board_reversed)  = board_reversed_opt {
        let moves2 = MoveGen::new_legal(&board_reversed).collect::<Vec<ChessMove>>();
        binding = [moves, &moves2].concat();
        total_moves = binding.as_slice();
    }
    else {
        total_moves = moves;
    }

    for p in pieces {
        let square = p.square_opt.unwrap();
        if (square.get_rank() == Rank::Fourth || square.get_rank() == Rank::Fifth)
        && (square.get_file() == File::D || square.get_file() == File::E)
        {
            res += match p.piece {
                Piece::Pawn => PAWN_VALUE,
                Piece::Knight => KNIGHT_VALUE,
                Piece::Bishop => BISHOP_VALUE,
                Piece::Rook => ROOK_VALUE,
                Piece::Queen => QUEEN_VALUE,
                Piece::King => KING_VALUE,
            } * match p.color {
                Color::Black => -1,
                Color::White => 1,
            };
        }   
    }

    for m in total_moves {
        if (m.get_dest().get_rank() == Rank::Fourth || m.get_dest().get_rank() == Rank::Fifth)
            && (m.get_dest().get_file() == File::D || m.get_dest().get_file() == File::E)
        {
            res += match board.piece_on(m.get_source()).unwrap() {
                Piece::Pawn => PAWN_VALUE,
                Piece::Knight => KNIGHT_VALUE,
                Piece::Bishop => BISHOP_VALUE,
                Piece::Rook => ROOK_VALUE,
                Piece::Queen => QUEEN_VALUE,
                Piece::King => KING_VALUE,
            } * match board.color_on(m.get_source()).unwrap() {
                Color::Black => -1,
                Color::White => 1,
            };
        }
    } 

    res
}

fn count_threats(board: &Board, pieces: &[PieceComplete], moves: &[ChessMove]) -> i32 {
    let mut res = 0;

    let board_reversed_opt = board.null_move();
    let total_moves: &[ChessMove];
    let binding: Vec<ChessMove>;
    if let Some(board_reversed) = board_reversed_opt {
        let moves2 = MoveGen::new_legal(&board_reversed).collect::<Vec<ChessMove>>();
        binding = [moves, &moves2].concat();
        total_moves = binding.as_slice();
    }
    else {
        total_moves = moves;
    }

    for attacked_piece in pieces {
        let attacked_square = attacked_piece.square_opt.unwrap();
        
        let attacking_value: i32 = total_moves
            .to_vec().iter()
            .filter(|m| m.get_dest() == attacked_square)
            .map(|m| match board.piece_on(m.get_source()) {
                Some(p) => match p {
                    Piece::Pawn => PAWN_VALUE,
                    Piece::Knight => KNIGHT_VALUE,
                    Piece::Bishop => BISHOP_VALUE,
                    Piece::Rook => ROOK_VALUE,
                    Piece::Queen => QUEEN_VALUE,
                    Piece::King => KING_VALUE,
                },
                None => 0,
            }).sum::<i32>();

        if attacking_value != 0 {
            let defense_value = count_defenses(board, pieces, attacked_piece);
            let attacked_value = match attacked_piece.piece {
                Piece::Pawn => PAWN_VALUE,
                Piece::Knight => KNIGHT_VALUE,
                Piece::Bishop => BISHOP_VALUE,
                Piece::Rook => ROOK_VALUE,
                Piece::Queen => QUEEN_VALUE,
                Piece::King => KING_VALUE,
            };

            let resultado_contienda = if defense_value == 0 {
                attacked_value
            } 
            else {
                attacked_value - attacking_value
            };
            
            let resultado_contienda = i32::max(0, resultado_contienda);

            res += match attacked_piece.color {
                Color::Black => resultado_contienda,
                Color::White => -resultado_contienda,
            }
        }    
    }

    res
}

fn count_defenses(board: &Board, pieces: &[PieceComplete], piece_dest: &PieceComplete) -> i32 {
    let mut res = 0;
    let casillas_ocupadas = board.color_combined(Color::White) | board.color_combined(Color::Black);
    let casilla_destino = piece_dest.square_opt.unwrap();

    for p in pieces
        .iter()
        .filter(|pc| pc.color == piece_dest.color && **pc != *piece_dest)
    {
        let mul;

        res += match p.piece {
            Piece::Pawn => {
                mul = PAWN_VALUE;
                chess::get_pawn_attacks(p.square_opt.unwrap(), p.color, casillas_ocupadas)
            }
            Piece::Knight => {
                mul = KNIGHT_VALUE;
                chess::get_knight_moves(p.square_opt.unwrap())
            }
            Piece::Bishop => {
                mul = BISHOP_VALUE;
                chess::get_bishop_moves(p.square_opt.unwrap(), casillas_ocupadas)
            }
            Piece::Rook => {
                mul = ROOK_VALUE;
                chess::get_rook_moves(p.square_opt.unwrap(), casillas_ocupadas)
            }
            Piece::Queen => {
                mul = QUEEN_VALUE;
                chess::get_rook_moves(p.square_opt.unwrap(), casillas_ocupadas)
                    | chess::get_bishop_moves(p.square_opt.unwrap(), casillas_ocupadas)
            }
            Piece::King => {
                mul = KING_VALUE;
                chess::get_king_moves(p.square_opt.unwrap())
            }
        }
        .filter(|s| *s == casilla_destino)
        .count() as i32
            * mul;
    }

    res
}

fn count_available_squares(board: &Board, pieces: &[PieceComplete]) -> i32 {
    let mut res = 0;
    let casillas_negras = board.color_combined(Color::Black);
    let casillas_blancas = board.color_combined(Color::White);
    let casillas_ocupadas = casillas_negras | casillas_blancas;

    for p in pieces {
        res += match p.piece {
            Piece::Pawn => chess::get_pawn_quiets(p.square_opt.unwrap(), p.color, casillas_ocupadas).count() as i32 + 
                chess::get_pawn_attacks(p.square_opt.unwrap(), p.color, match p.color {
                    Color::Black => *casillas_blancas,
                    Color::White => *casillas_negras,
                }).count() as i32,
            //chess::get_pawn_moves(p.square_opt.unwrap(), p.color, casillas_ocupadas).count() as i32,
            Piece::Knight  => chess::get_knight_moves(p.square_opt.unwrap())
                .filter(|sq| board.color_on(*sq).is_none() || board.color_on(*sq).unwrap() != p.color)
                .count() as i32,
            Piece::Bishop => chess::get_bishop_moves(p.square_opt.unwrap(), casillas_ocupadas)
                .filter(|sq| board.color_on(*sq).is_none() || board.color_on(*sq).unwrap() != p.color)
                .count() as i32,
            Piece::Rook => chess::get_rook_moves(p.square_opt.unwrap(), casillas_ocupadas)
                .filter(|sq| board.color_on(*sq).is_none() || board.color_on(*sq).unwrap() != p.color)
                .count() as i32,
            Piece::Queen => (chess::get_bishop_moves(p.square_opt.unwrap(), casillas_ocupadas)
                .filter(|sq| board.color_on(*sq).is_none() || board.color_on(*sq).unwrap() != p.color)
                .count() 
                + chess::get_rook_moves(p.square_opt.unwrap(), casillas_ocupadas).filter(|sq| board.color_on(*sq).is_none() || board.color_on(*sq).unwrap() != p.color).count())
                as i32,
            Piece::King => chess::get_king_moves(p.square_opt.unwrap())
                .filter(|sq| board.color_on(*sq).is_none() || board.color_on(*sq).unwrap() != p.color)
                .count() as i32
        } * match p.color {
            Color::Black => -1,
            Color::White => 1,
        };
    }
    res
}

fn king_position(board: &Board, pieces: &[PieceComplete]) -> i32 {
    let mut res = 0;

    // Finales
    if pieces.len() <= 20 {
        let king_square = board.king_square(Color::Black);
        let king_rank = king_square.get_rank().to_index();
        let king_file = king_square.get_file().to_index();

        res += -KING_PROXIMITY_CENTER[king_rank][king_file];

        let king_square = board.king_square(Color::White);
        let king_rank = king_square.get_rank().to_index();
        let king_file = king_square.get_file().to_index();

        res += KING_PROXIMITY_CENTER[king_rank][king_file];
    } 
    else {
        let king_square = board.king_square(Color::Black);
        let king_rank = king_square.get_rank();

        if king_rank == Rank::Eighth {
            res -= 50;
        }
        
        let king_square = board.king_square(Color::White);
        let king_rank = king_square.get_rank();
        
        if king_rank == Rank::First {
            res += 50;
        }
    }

    res
}

fn count_pieces(pieces: &[PieceComplete]) -> i32 {
    let mut res = 0;

    for p in pieces.iter() {
        res += piece_value(&p);
    }

    res
}

fn get_pieces(board: &Board) -> Vec<PieceComplete> {
    let mut res: Vec<PieceComplete> = Vec::new();

    for rank in 0..=7 {
        for file in 0..=7 {
            let square = Square::make_square(Rank::from_index(rank), File::from_index(file));
            let piece = board.piece_on(square);

            match piece {
                Some(p) => res.push(PieceComplete {
                    piece: p,
                    color: board.color_on(square).unwrap(),
                    square_opt: Some(square),
                }),
                None => (),
            }
        }
    }

    res
}

fn piece_value(piece: &PieceComplete) -> i32 {
    let v = match piece.piece {
        Piece::Pawn => PAWN_VALUE,
        Piece::Knight => KNIGHT_VALUE,
        Piece::Bishop => BISHOP_VALUE,
        Piece::Rook => ROOK_VALUE,
        Piece::Queen => QUEEN_VALUE,
        Piece::King => KING_VALUE,
    };

    match piece.color {
        Color::Black => -v,
        Color::White => v,
    }
}
