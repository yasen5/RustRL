use lunar_lander_rl::{game, model, train};
use macroquad::{input::{KeyCode, is_key_down}, window::Conf};

#[macroquad::main(window_conf)]
async fn main() {
    let mut agent: model::Model = model::Model::new(4);
    agent.add_layer(5, 256);
    agent.add_layer(256, 64);
    agent.add_layer(64, 4);
    let mut new_game: game::Game = game::Game::new();
    train::train(&mut new_game, &mut agent).await;
}

fn choose() -> usize {
    if is_key_down(KeyCode::A) {
        1
    } else if is_key_down(KeyCode::D) {
        0
    } else if is_key_down(KeyCode::W) {
        2
    }
    else {
        3
    }
}

pub fn window_conf() -> Conf {
    Conf {
        window_title: "Lunar Lander".to_string(),
        window_width: 800,
        window_height: 800,
        fullscreen: false,
        ..Default::default()
    }
}
