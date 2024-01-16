use std::fs::File;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Registro {
    game: u32,
    step: u32,
    td_target: f64,
    q_values_action: f64,
    loss: f64,
    epsilon: f64,
    steps_secs: f32,
}

pub fn write_to_csv(game: u32, step: u32, td_target: f64, q_values_action: f64, loss: f64, epsilon: f64, steps_secs: f32, first_log: bool) {
    let registro = Registro {
        game: game,
        step: step,
        td_target: td_target,
        q_values_action: q_values_action,
        epsilon: epsilon,
        steps_secs: steps_secs,    
        loss: loss,
    };

    let file = File::options().append(true).create(true).open("registro.csv").unwrap();

    let mut wtr = csv::WriterBuilder::new().delimiter(b';').has_headers(false).from_writer(file);

    if first_log {
        wtr.write_record(&["game", "step", "td_target", "q_values_action", "loss", "epsilon", "steps_secs"]).unwrap();
    }

    wtr.serialize(registro).unwrap();

    wtr.flush().unwrap();
}
