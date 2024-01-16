use std::{io, path::Path, str::FromStr, time::Instant};

use chess::{Board, ChessMove, Color};
use crossterm::{cursor::MoveTo, execute};
use rand::Rng;
use tch::{
    nn::{self, Module, OptimizerConfig},
    no_grad, Tensor,
};

use crate::{
    board_controller::{board_to_tensor, display_board},
    csv_manager::write_to_csv,
    dqn_nn_model::DQNModelNN,
    evaluator::evaluate,
    game_manager::GameManager,
    replay_buffer::{BufferElement, ReplayBuffer},
};

pub fn dqn_train() {
    let ruta_nn = "nn.pth";
    let load_path = Path::new(ruta_nn);

    let num_games = 10000;

    // Calculo de epsilon
    let init_epsilon: f64 = 1.0;
    let final_epsilon: f64 = 0.01;

    // replayBuffer
    let capacity: usize = 500000;
    let mut buffer = ReplayBuffer::new(capacity);
    let gamma = 0.99;

    // entrenamiento de target network
    let valor_minimo_entrenar = 35000;
    let frecuencia_entrenamiento = 10;

    let samples_size: usize = 16;

    // Redes neuronales
    let mut vs_q_network = nn::VarStore::new(tch::Device::Mps);
    let q_network = DQNModelNN::new(&vs_q_network.root());
    let mut vs_target_network = nn::VarStore::new(tch::Device::Mps);
    let target_network = DQNModelNN::new(&vs_target_network.root());

    if load_path.exists() {
        vs_q_network.load(load_path).unwrap();
    }

    let learning_rate = 0.001;
    let weight_decay = 0.01;
    //let mut optimizador = tch::nn::RmsProp::default().build(&vs_q_network, learning_rate).unwrap();
    let mut optimizador = tch::nn::Adam::default()
        .build(&vs_q_network, learning_rate)
        .unwrap();
    optimizador.set_weight_decay(weight_decay);
    let target_network_update_freq = 10000;

    let start_time = Instant::now();
    let mut first_log = true;

    _ = vs_target_network.copy(&vs_q_network).unwrap();

    let mut rng = rand::thread_rng();
    let mut total_cont: u32 = 0;

    for game_step in 0..num_games {
        let mut game_manager = GameManager::new();
        let mut game_result: Option<chess::GameResult> = game_manager.result();
        let mut game_cont: u32 = 0;
        let mut force_random = false;

        // Para log
        let mut random_moves = 0;

        let epsilon = get_epsilon(game_step, init_epsilon, final_epsilon, num_games);

        while game_result.is_none() {
            let mut moves = game_manager.get_moves();
            let mut action_index  = 127;

            if rng.gen::<f64>() < epsilon || force_random {
                action_index = rng.gen_range(0..moves.len());
                random_moves += 1;
            } 
            else {
                let qv = q_network.forward(&board_to_tensor(&game_manager.board()));
                let filtered_output = qv.softmax(-1, tch::Kind::Float);
                action_index = filtered_output.argmax(None, false).int64_value(&[]) as usize;    
            }

            let movimiento_opt: Option<ChessMove> = moves.nth(action_index);
            let actual_board = game_manager.board();

            if let Some(movimiento) = movimiento_opt {
                game_manager.do_move(movimiento);
                game_manager.declare_draw();
                let next_board = game_manager.board();

                game_result = game_manager.result();

                let buff_el = BufferElement {
                    actual_state: actual_board.to_string(),
                    action: action_index,
                    reward: evaluate(&next_board, game_cont, game_result)
                        * match actual_board.side_to_move() {
                            Color::Black => -1.0,
                            Color::White => 1.0,
                        },
                    next_state: next_board.to_string(),
                    done: game_result.is_some(),
                };

                buffer.add(buff_el);

                game_cont += 1;
                force_random = false;
            } 
            else {
                let buff_el = BufferElement {
                    actual_state: actual_board.to_string(),
                    action: action_index,
                    reward: match actual_board.side_to_move() {
                        Color::Black => 1.0,
                        Color::White => -1.0,
                    },
                    next_state: actual_board.to_string(),
                    done: true,
                };

                buffer.add(buff_el);
                force_random = true;
            }

            total_cont += 1;

            if total_cont > valor_minimo_entrenar && total_cont % frecuencia_entrenamiento == 0
            {
                let samples = buffer.get_samples(samples_size);

                let mut actual_states: Vec<Tensor> = Vec::new();
                let mut next_states: Vec<Tensor> = Vec::new();
                let mut rewards: Vec<f32> = Vec::new();
                let mut dones: Vec<bool> = Vec::new();
                let mut actions: Vec<i32> = Vec::new();

                for sample in samples {
                    actual_states.push(board_to_tensor(
                        &Board::from_str(&sample.actual_state).unwrap(),
                    ));
                    next_states.push(board_to_tensor(
                        &Board::from_str(&sample.next_state).unwrap(),
                    ));
                    rewards.push(sample.reward);
                    dones.push(sample.done);
                    actions.push(sample.action as i32);
                }

                let actual_states_tensor = Tensor::stack(actual_states.as_slice(), 0);
                let next_states_tensor = Tensor::stack(next_states.as_slice(), 0);
                let rewards_tensor = Tensor::from_slice(rewards.as_slice()).to_device(tch::Device::Mps);
                let dones_tensor = Tensor::from_slice(dones.as_slice()).to(tch::Device::Mps);

                let mut td_target: Tensor =
                    Tensor::zeros(samples_size as i64, tch::kind::DOUBLE_CPU);
                no_grad(|| {
                    let (target_max, _) = target_network
                        .forward(&next_states_tensor)
                        .max_dim(1, false);

                    td_target = (rewards_tensor + (gamma as f64 * target_max * dones_tensor.logical_not())).to(tch::Device::Mps);
                });

                let q_values: Tensor = q_network.forward(&actual_states_tensor).to(tch::Device::Mps);
                let actions_tensor =
                    Tensor::from_slice(actions.as_slice()).to_kind(tch::Kind::Int64).to_device(tch::Device::Mps);

                let q_values_action = q_values.gather(1, &actions_tensor.unsqueeze(-1), false);
                //let loss = td_target.huber_loss(&q_values_action.squeeze_dim(1), tch::Reduction::Mean, 1.0);
                //let loss = q_values_action.squeeze_dim(1).mse_loss(&td_target, tch::Reduction::Mean);
                let loss = q_values_action.squeeze_dim(1).smooth_l1_loss(&td_target, tch::Reduction::Mean, 1.0);
                let loss_value = loss.double_value(&[]);

                //optimizador.zero_grad();
                optimizador.backward_step(&loss);
                
                //optimizador.backward_step_clip(&loss, 10.0);

                if total_cont % 1000 == 0 {
                    let td_target_mean = td_target.to_device(tch::Device::Cpu).mean(tch::Kind::Double).double_value(&[]);
                    let q_values_action_mean =
                        q_values_action.to_device(tch::Device::Cpu).mean(tch::Kind::Double).double_value(&[]);
                    let steps_secs = total_cont as f32 / start_time.elapsed().as_secs_f32();

                    write_log(
                        loss_value,
                        td_target_mean,
                        q_values_action_mean,
                        steps_secs,
                    );

                    write_to_csv(
                        game_step,
                        total_cont,
                        td_target_mean,
                        q_values_action_mean,
                        loss_value,
                        epsilon,
                        steps_secs,
                        first_log,
                    );
                    first_log = false;
                }
            }

            if total_cont % target_network_update_freq == 0 {
                _ = vs_target_network.copy(&vs_q_network).unwrap();
                vs_q_network.save(ruta_nn).unwrap();
            }
        }

        display_board(&game_manager.board());
        _ = execute!(io::stdout(), MoveTo(0, 11));
        println!(
            "Game Result {:?} - cont: {}",
            game_result.unwrap(),
            game_cont
        );
        _ = execute!(io::stdout(), MoveTo(0, 12));
        println!(
            "Exploration: {random_moves} - Exploitation: {}",
            game_cont - random_moves
        );
        //println!("{}", game_manager.get_historic());
    }
}

fn write_log(
    loss_value: f64,
    td_target_mean: f64,
    q_values_action_mean: f64,
    steps_secs: f32,
) {
    _ = execute!(io::stdout(), MoveTo(0, 13));
    println!("Loss: {}", loss_value);
    _ = execute!(io::stdout(), MoveTo(0, 14));
    println!("td target: {}", td_target_mean);
    _ = execute!(io::stdout(), MoveTo(0, 15));
    println!("q values action: {}", q_values_action_mean);
    _ = execute!(io::stdout(), MoveTo(0, 16));
    println!("Steps/secs: {:?}", steps_secs);
}

fn get_epsilon(step: u32, start_v: f64, end_v: f64, total_steps: u32) -> f64 {
    if step > total_steps {
        return end_v;
    }

    return start_v + (end_v - start_v) * (step as f64 / total_steps as f64);
}
