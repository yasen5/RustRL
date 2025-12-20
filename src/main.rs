use std::{io, thread, time::Duration};

use macroquad::{input::{KeyCode, is_key_down}, window::{Conf, next_frame}};

mod model;
mod game;

#[macroquad::main(window_conf)]
async fn main() {
    println!("Hello, world!");
    actual_main().await;
}

async fn actual_main() {
    let mut newGame = game::Game::new();
    loop {
        let mut choice: u8 = 3;
        if is_key_down(KeyCode::A) {
            choice = 1;
        }
        else if is_key_down(KeyCode::D) {
            choice = 0;
        }
        else if is_key_down(KeyCode::W) {
            choice = 2;
        }
        newGame.step(choice);
        newGame.draw(choice);
        thread::sleep(Duration::from_millis(100));
        next_frame().await;
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Lunar Lander".to_string(),
        window_width: 800,
        window_height: 800,
        fullscreen: false,
        ..Default::default()
    }
}