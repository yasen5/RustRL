use lunar_lander_rl::game;
use macroquad::window::Conf;

#[macroquad::main(window_conf)]
async fn main() {
    game::run_game().await;
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