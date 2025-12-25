use lunar_lander_rl::{game, model};
use ndarray::{Array, Array1};
fn main() {
    let mut score: i16 = 0;
    let mut state: Array1<f32> = Array::from_elem(5, 0.);
    let mut agent: model::Model = model::Model::new(4);
    agent.add_layer(Box::new(model::LinearLayer::new(5, 256)));
    agent.add_layer(Box::new(model::LinearLayer::new(256, 64)));
    agent.add_layer(Box::new(model::LinearLayer::new(64, 4)));
    let mut new_game: game::Game = game::Game::new();
    for _ in 0..10000000 {
        new_game.state().to_vec(&mut state);
        let choice = agent
            .forward(&state)
            .indexed_iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let outcome: game::StepOutcome = new_game.step(choice);
        score += outcome.score;
        if outcome.finished {
            new_game.reset();
        }
    }
}
