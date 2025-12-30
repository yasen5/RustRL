use ndarray::Array1;
use rand::{Rng};

use crate::game;

const SESSIONS: u16 = 200;
const ITER_DISPLAY_PRECISION: u16 = 20;
const LOG_INTERVAL: u16 = SESSIONS / ITER_DISPLAY_PRECISION;
const GAMMA: f32 = 0.99;
const LEARNING_RATE: f32 = 0.00001;
const OBSERVATION_SPACE: usize = 7;

pub async fn train(game: &mut crate::game::Game, agent: &mut crate::model::Model) {
    let mut acted_upon_state: Array1<f32>;
    let mut target: crate::model::Model = agent.clone();
    let mut state: Array1<f32> = Array1::zeros(OBSERVATION_SPACE);
    let mut loss_derivative;
    let mut actions: [u16; 6] = [0; 6];
    let mut epsilon: f32 = 1.;
    let mut rng = rand::rng();
    for iter in 0..SESSIONS {
        epsilon *= 9. / 10.;
        let mut score: i16 = 0;
        loop {
            loss_derivative = Array1::zeros(6);
            game.state().to_vec(&mut state);
            acted_upon_state = state.clone();
            let output = agent.forward(&state).clone();
            let choice: usize = if rng.random::<f32>() > epsilon {
                output
                    .indexed_iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap()
            } else {
                rng.random_range(0..6)
            };
            actions[choice] += 1;
            let (reward, finished) = game.step(choice, false);
            game.state().to_vec(&mut state);
            if finished {
                loss_derivative[choice] = output[choice] - reward as f32;
            } else {
                let target_output = target.forward(&state);
                let next_state_value_estimate: f32 = target_output
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .map(|v| *v)
                    .unwrap();
                loss_derivative[choice] =
                    output[choice] - (reward as f32 + GAMMA * next_state_value_estimate);
            }
            agent.backprop(&acted_upon_state, &mut loss_derivative);
            score += reward;
            if finished {
                display_progress(iter);
                agent.apply_gradients(LEARNING_RATE, game.steps);
                println!(
                    "Score: {}\t Action Count: {:?}\tOutput: {:?}",
                    score, actions, output
                );
                game.reset();
                break;
            }
        }
        if iter % 4 == 0 {
            target = agent.clone();
        }
    }
    game::run_game(
        || {
            game.state().to_vec(&mut state);
            agent
                .forward(&state)
                .indexed_iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        },
        true,
    )
    .await;
}

fn display_progress(iter: u16) {
    if LOG_INTERVAL == 0 || iter % LOG_INTERVAL != 0 {
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
