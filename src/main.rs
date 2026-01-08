use lunar_lander_rl::{game, model, train};
use macroquad::window::Conf;

const HUMAN_PLAYER: bool = false;

#[macroquad::main(window_conf)]
async fn main() {
    let mut game: game::Game = game::Game::new();
    if HUMAN_PLAYER {
        game::run_game(choose).await;
    } else {
        let mut agent: model::Model = model::Model::new();
        agent.add_layer(game.observation_space, 512, true);
        agent.add_layer(512, 256, true);
        agent.add_layer(256, 64, true);
        agent.add_layer(64, game.action_space, false);
        train::train(&mut game, &mut agent).await;
    }
}

fn choose() -> usize {
    0
    // if is_key_down(KeyCode::A) {
    //     if is_key_down(KeyCode::W) { 5 } else { 1 }
    // } else if is_key_down(KeyCode::D) {
    //     if is_key_down(KeyCode::W) { 4 } else { 0 }
    // } else if is_key_down(KeyCode::W) {
    //     2
    // } else {
    //     3
    // }
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
