mod model;
mod game;

fn main() {
    println!("Hello, world!");
    let mut newGame = game::Game::new();
    newGame.step(0);
}
