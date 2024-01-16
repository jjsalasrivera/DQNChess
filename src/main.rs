mod game_manager;
mod evaluator;
mod dqn_train;
mod board_controller;
mod replay_buffer;
mod dqn_nn_model;
mod csv_manager;
mod humano_vs_ia;

use std::io::{self, Write};

use crossterm::{
    cursor::{EnableBlinking, Hide, MoveTo, MoveToNextLine, Show},
    event::{read, KeyCode},
    execute,
    style::{Color, SetForegroundColor},
    terminal::{self, Clear, ClearType},
};

fn main() {
    _ = terminal::enable_raw_mode();

    let opciones = vec![
        "Entrenar Maquina",
        "Humano vs Maquina",
        "Maquina vs Humano",
        "Humano vs Humano"
    ];

    let mut seleccionado = 0;

    loop {
        _ = print_menu(&opciones, seleccionado);

        if let Ok(key_event) = read() {
            match key_event {
                crossterm::event::Event::Key(crossterm::event::KeyEvent {
                    code,
                    kind: _,
                    state: _,
                    modifiers: _,
                }) => match code {
                    KeyCode::Up => {
                        if seleccionado > 0 {
                            seleccionado -= 1;
                        } else {
                            seleccionado = opciones.len() - 1
                        }
                    }
                    KeyCode::Down => {
                        if seleccionado < opciones.len() - 1 {
                            seleccionado += 1;
                        } else {
                            seleccionado = 0;
                        }
                    }
                    KeyCode::Enter => {
                        _ = execute!(io::stdout(), Clear(ClearType::All), MoveTo(0, 0));
                        match seleccionado {
                            0 => { dqn_train::dqn_train() }
                            1 => { humano_vs_ia::jugar(chess::Color::White) }
                            2 => { humano_vs_ia::jugar(chess::Color::Black) }
                            3 => { println!("No implementado") }
                            _ => {}
                        }
                        break;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
    _ = execute!(io::stdout(), Show, EnableBlinking, MoveToNextLine(1));
    _ = terminal::disable_raw_mode();
}

fn print_menu(items: &[&str], selected_index: usize) -> io::Result<()> {
    execute!(io::stdout(), Clear(ClearType::All), MoveTo(0, 0), Hide)?;

    for (i, item) in items.iter().enumerate() {
        execute!(io::stdout(), MoveTo(0, i as u16))?;
        if i == selected_index {
            execute!(io::stdout(), SetForegroundColor(Color::Yellow))?;
        }
        println!("{}", item);
        if i == selected_index {
            execute!(io::stdout(), SetForegroundColor(Color::Reset))?;
        }
    }

    io::stdout().flush()?;
    Ok(())
}
