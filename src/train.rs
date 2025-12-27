use ndarray::{Array, Array1};

const SESSIONS: u16 = 40;
const ITER_DISPLAY_PRECISION: u16 = 20;
const LOG_INTERVAL: u16 = SESSIONS / ITER_DISPLAY_PRECISION;

pub fn fake_train(game: &mut crate::game::Game, agent: &mut crate::model::Model) {
    let mut target: crate::model::Model = agent.clone();
    for iter in 0..SESSIONS {
        let mut score: i16 = 0;
        let mut state: Array1<f32> = Array::from_elem(game.observation_space, 0.);
        let mut batch_gradients: Array1<f32> = Array::from_elem(game.action_space, 0.);
        game.state().to_vec(&mut state);
        let current_state = state.clone();
        let (choice, reward_prediction): (usize, &f32) = agent
            .forward(&state)
            .indexed_iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (reward, finished) = game.step(choice);
        game.state().to_vec(&mut state);
        batch_gradients[choice] = (reward as f32
            + *target
                .forward(&state)
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.))
            - *reward_prediction;
        score += reward;
        if finished {
            game.reset();
            batch_gradients /= game.steps as f32;
            agent.backprop(state, &mut batch_gradients);
        }
        display_progress(iter);
    }
}
pub fn train(game: &mut crate::game::Game, agent: &mut crate::model::Model) {
    let mut target: crate::model::Model = agent.clone();
    for iter in 0..SESSIONS {
        let mut score: i16 = 0;
        let mut state: Array1<f32> = Array::from_elem(game.observation_space, 0.);
        let mut batch_gradients: Array1<f32> = Array::from_elem(game.action_space, 0.);
        game.state().to_vec(&mut state);
        let current_state = state.clone();
        let (choice, reward_prediction): (usize, &f32) = agent
            .forward(&state)
            .indexed_iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (reward, finished) = game.step(choice);
        game.state().to_vec(&mut state);
        batch_gradients[choice] = (reward as f32
            + *target
                .forward(&state)
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.))
            - *reward_prediction;
        score += reward;
        if finished {
            game.reset();
            batch_gradients /= game.steps as f32;
            agent.backprop(state, &mut batch_gradients);
        }
        display_progress(iter);
    }
}

fn display_progress(iter: u16) {
    if iter % LOG_INTERVAL != 0 {
        return;
    }
    let hashtags: u16 = iter / LOG_INTERVAL;
    let spaces: u16 = ITER_DISPLAY_PRECISION - hashtags;
    print!("[");
    for _ in 0..hashtags {
        print!("#");
    }
    for _ in 0..spaces {
        print!(" ");
    }
    println!("]");
}
