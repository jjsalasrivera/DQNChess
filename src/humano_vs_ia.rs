use std::{io::{self}, path::Path, process::exit};

use chess::{Color, ChessMove};
use crossterm::{execute, cursor::{MoveTo, Show, EnableBlinking}, style::Print, terminal};
use tch::{nn::{self, Module}, no_grad, Tensor};

use crate::{game_manager::GameManager, dqn_nn_model::DQNModelNN, board_controller::{display_board, board_to_tensor}, evaluator::evaluate};

pub fn jugar(human_side: Color) {
    let ruta_nn = "nn.pth";
    let load_path = Path::new(ruta_nn);

    let mut vs = nn::VarStore::new(tch::Device::Mps);
    let q_network = DQNModelNN::new(&vs.root());

    if load_path.exists() {
        vs.load(load_path).unwrap();
    }
    else {
        println!("No se encontro la red neuronal");
        exit(1);
    }
   
    _ = execute!(io::stdout(), Show, EnableBlinking);

    let mut game_manager = GameManager::new();
    let mut game_result: Option<chess::GameResult> = game_manager.result();
    let mut moves_cont = 0;

    while game_manager.declare_draw() || game_result.is_none() {
        moves_cont += 1;
        if game_manager.side_to_move() == human_side {
            let board = game_manager.board();
            display_board(&board);
            let valor = evaluate(&board, moves_cont, game_manager.result());
            _ = execute!(io::stdout(), MoveTo(0, 15));
            print!("Valor: {}", valor);
            human_make_move(&mut game_manager);
            let valor = evaluate(&game_manager.board(), moves_cont, game_manager.result());
            _ = execute!(io::stdout(), MoveTo(0, 15));
            print!("Valor: {}", valor);
        } 
        else {
            ia_make_move(&mut game_manager, &q_network);
        }

        game_result = game_manager.result();
    }

    _ = execute!(io::stdout(), MoveTo(0, 11), Print("Juego terminado\n"));
    _ = execute!(io::stdout(), MoveTo(0, 12), Print("Resultado: "));
    _ = execute!(io::stdout(), MoveTo(0, 13), Print(format!("{:?}\n", game_result.unwrap())));
    _ = execute!(io::stdout(), MoveTo(0, 14), Print("Pulse ENTER para continuar..."));
    io::stdin().read_line(&mut String::new()).unwrap();
}

fn ia_make_move(game_manager: &mut GameManager, q_network: &DQNModelNN) {
    let moves = game_manager.get_moves();
    let moves_vec = moves.into_iter().collect::<Vec<ChessMove>>();

    let mut filtered_output: Tensor = Default::default();
    no_grad(|| {
        let qv = q_network.forward(&board_to_tensor(&game_manager.board()));
        //let softmax = qv.softmax(0, tch::Kind::Float);
        //filtered_output = softmax.argsort(-1,true);
        filtered_output = qv.argsort(-1,true);
    });

    let mut correct_index = false;
    let mut i = 0;

    println!("Movimientos posibles: {:?}", moves_vec.len());

    while !correct_index {
        let action_index = filtered_output.int64_value(&[0, i]) as usize;
        let movimiento_opt = moves_vec.get(action_index);

        if let Some(movimiento) = movimiento_opt {
            let (move_is_good, _game_res_opt)  = game_manager.do_move(*movimiento);

            if move_is_good {
                correct_index = true;
            }
        }
        i += 1;
    }
}

fn human_make_move(game_manager: &mut GameManager) {
    let move_str = get_move_str();
    let chess_move_res = ChessMove::from_san(&game_manager.board(), &move_str.trim());

    if chess_move_res.is_ok() {
        let chess_move = chess_move_res.unwrap();
        let (move_is_good, _game_res_opt)  = game_manager.do_move(chess_move);

        if !move_is_good {
            _ = execute!(io::stdout(), MoveTo(0, 12));
            println!("Movimiento invalido");
        }
    }
    else {
        _ = execute!(io::stdout(), MoveTo(0, 13));
        println!("Movimiento invalido");
    }
}

fn get_move_str() -> String {
    _ = execute!(io::stdout(), MoveTo(0, 11), Print("Inserte movimiento: ".to_string()));

    _ = terminal::disable_raw_mode();
    let mut movida_str = String::new();
    io::stdin().read_line(&mut movida_str).unwrap();
    _ = terminal::enable_raw_mode();

    movida_str
}