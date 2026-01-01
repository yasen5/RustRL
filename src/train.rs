use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::{Rng, SeedableRng};

use crate::game;

const SESSIONS: u16 = 2;
const ITER_DISPLAY_PRECISION: u16 = 20;
const LOG_INTERVAL: u16 = SESSIONS / ITER_DISPLAY_PRECISION;
const GAMMA: f32 = 0.99;
const LEARNING_RATE: f32 = 0.001;
const OBSERVATION_SPACE: usize = 2;
const ACTION_SPACE: usize = 2;
const BATCH_SIZE: u16 = 4;
const EPSILON_DECAY: f32 = 0.95;
pub const SEED: u64 = 42;


pub fn fake_train(agent: &mut crate::model::Model) {
    let mut rng: rand::rngs::StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let state: Array1<f32> = Array1::random(
        OBSERVATION_SPACE,
        rand::distr::Uniform::new_inclusive(-1.0, 1.0).unwrap(),
    );
    let mut loss_derivative;
    let mut actions: [u16; ACTION_SPACE] = [0; ACTION_SPACE];
    let epsilon: f32 = 1.;
    let mut steps: u16 = 0;
    for iter in 0..SESSIONS {
        steps += 1;
        loss_derivative = Array1::zeros(ACTION_SPACE);
        let agent_prediction = agent.forward(&state).clone();
        let choice: usize = if rng.random::<f32>() > epsilon {
            agent_prediction
                .indexed_iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        } else {
            rng.random_range(0..ACTION_SPACE)
        };
        actions[choice] += 1;
        let reward: i16 = choice as i16;
        loss_derivative[choice] = agent_prediction[choice] - (reward as f32);
        agent.backprop(&state, &mut loss_derivative);
        if steps % BATCH_SIZE == 0 {
            agent.apply_gradients(LEARNING_RATE, BATCH_SIZE);
        }
        display_progress(iter);
    }
}

pub async fn train(game: &mut crate::game::Game, agent: &mut crate::model::Model) {
    let mut rng: rand::rngs::StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut acted_upon_state: Array1<f32>;
    let mut target: crate::model::Model = agent.clone();
    let mut state: Array1<f32> = Array1::zeros(OBSERVATION_SPACE);
    let mut loss_derivative;
    let mut epsilon: f32 = 0.0;
    let mut residual = 0;
    let mut errors: [(usize, f32); 15] = [(0, 0.); 15];
    let mut target_preds: [(usize, f32); 15] = [(0, -5.); 15];
    let mut preds: [(usize, f32); 15] = [(0, 0.); 15];
    for iter in 0..SESSIONS {
        epsilon *= EPSILON_DECAY;
        let mut rewards_per_action: [f32; ACTION_SPACE] = [0.; ACTION_SPACE];
        let mut actions: [u16; ACTION_SPACE] = [0; ACTION_SPACE];
        let mut intentional_actions: [u16; ACTION_SPACE] = [0; ACTION_SPACE];
        let mut score: f32 = 0.;
        loop {
            loss_derivative = Array1::zeros(ACTION_SPACE);
            game.state().to_vec(&mut state);
            acted_upon_state = state.clone();
            let agent_prediction = agent.forward(&state).clone();
            dbg!(&agent_prediction);
            let choice: usize = if rng.random::<f32>() > epsilon {
                let temp_choice = agent_prediction
                    .indexed_iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                intentional_actions[temp_choice] += 1;
                temp_choice
            } else {
                rng.random_range(0..ACTION_SPACE)
            };
            let (reward, finished) = game.step(choice, false);
            rewards_per_action[choice] += reward;
            preds[game.steps as usize - 1] = (choice, agent_prediction[choice]);
            game.state().to_vec(&mut state);
            if finished {
                loss_derivative[choice] = agent_prediction[choice] - reward as f32;
            } else {
                let target_prediction = target.forward(&state);
                let next_state_choice: usize = target_prediction
                    .indexed_iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                let next_state_value_estimate: f32 = target_prediction[next_state_choice];
                if game.steps % 10 == 0 {
                    println!(
                        "Next value estimate: {:?}\tReward: {:?}\nQ-prediction:{:?}",
                        next_state_value_estimate, reward, agent_prediction
                    );
                }
                loss_derivative[choice] =
                    agent_prediction[choice] - (reward as f32 + GAMMA * next_state_value_estimate);
                target_preds[game.steps as usize - 1] =
                    (next_state_choice, next_state_value_estimate);
                dbg!(next_state_value_estimate);
            }
            actions[choice] += 1;
            errors[game.steps as usize - 1] = (choice, loss_derivative[choice]);
            // println!(
            //     "Score: {}\tIntentional Action Count: {:?}\tFull Actions: {:?}\tRewards: {:?}\tOutput: {:?}",
            //     score, intentional_actions, actions, rewards_per_action, agent_prediction
            // );
            agent.backprop(&acted_upon_state, &mut loss_derivative);
            if (game.steps + residual) % BATCH_SIZE == 0 {
                residual = 0;
                println!("**************************************************");
                agent.apply_gradients(LEARNING_RATE, BATCH_SIZE);
            }
            score += reward;
            if finished {
                residual += game.steps % BATCH_SIZE;
                for i in 0..OBSERVATION_SPACE {
                    rewards_per_action[i] = if intentional_actions[i] == 0 {
                        0.
                    } else {
                        rewards_per_action[i] / intentional_actions[i] as f32
                    };
                }
                println!(
                    "Score: {}\tIntentional Action Count: {:?}\tFull Actions: {:?}\tRewards: {:?}\tOutput: {:?}",
                    score, intentional_actions, actions, rewards_per_action, agent_prediction
                );
                if iter % 2 == 0 {
                    target = agent.clone();
                }
                game.reset();
                break;
            }
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
