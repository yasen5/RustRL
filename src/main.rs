use lunar_lander_rl::{game, model, train};
fn main() {
    let mut agent: model::Model = model::Model::new(4);
    agent.add_layer(Box::new(model::LinearLayer::new(5, 256)));
    agent.add_layer(Box::new(model::LinearLayer::new(256, 64)));
    agent.add_layer(Box::new(model::LinearLayer::new(64, 4)));
    let mut new_game: game::Game = game::Game::new();
    train::train(&mut new_game, &mut agent);
}
