use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;

use crate::game;

const SESSIONS: u16 = 5;
const ITER_DISPLAY_PRECISION: u16 = 20;
const LOG_INTERVAL: u16 = SESSIONS / ITER_DISPLAY_PRECISION;
const GAMMA: f32 = 0.99;
const LEARNING_RATE: f32 = 10000.;
const OBSERVATION_SPACE: usize = 2;
const ACTION_SPACE: usize = 2;
const BATCH_SIZE: u16 = 32;
const EPSILON_DECAY: f32 = 0.95;

pub fn fake_train(agent: &mut crate::model::Model) {
    let state: Array1<f32> = Array1::random(
        OBSERVATION_SPACE,
        rand::distr::Uniform::new_inclusive(-1.0, 1.0).unwrap(),
    );
    let mut loss_derivative;
    let mut actions: [u16; ACTION_SPACE] = [0; ACTION_SPACE];
    let epsilon: f32 = 1.;
    let mut rng = rand::rng();
    let mut steps: u16 = 0;
    for iter in 0..SESSIONS {
        steps += 1;
        loss_derivative = Array1::zeros(ACTION_SPACE);
        let output = agent.forward(&state).clone();
        let choice: usize = if rng.random::<f32>() > epsilon {
            output
                .indexed_iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        } else {
            rng.random_range(0..ACTION_SPACE)
        };
        actions[choice] += 1;
        let reward: i16 = choice as i16;
        loss_derivative[choice] = output[choice] - (reward as f32);
        agent.backprop(&state, &mut loss_derivative);
        if steps % BATCH_SIZE == 0 {
            agent.apply_gradients(LEARNING_RATE, BATCH_SIZE);
            println!("Action Count: {:?}\tOutput: {:?}", actions, output);
        }
        display_progress(iter);
    }
}

pub async fn train(game: &mut crate::game::Game, agent: &mut crate::model::Model) {
    let mut acted_upon_state: Array1<f32>;
    let mut target: crate::model::Model = agent.clone();
    let mut state: Array1<f32> = Array1::zeros(OBSERVATION_SPACE);
    let mut loss_derivative;
    let mut epsilon: f32 = 1.;
    let mut rng = rand::rng();
    let mut residual = 0;
    let mut prev_score: i16 = -10000;
    let mut rewards: [i16; 15] = [0; 15];
    let mut prev_rewards: [i16; 15] = [0; 15];
    let mut prev_errors: [(usize, f32); 15] = [(0, 0.); 15];
    let mut errors: [(usize, f32); 15] = [(0, 0.); 15];
    let mut prev_target_preds: [(usize, f32); 15] = [(0, 0.); 15];
    let mut target_preds: [(usize, f32); 15] = [(0, -5.); 15];
    let mut prev_preds: [(usize, f32); 15] = [(0, 0.); 15];
    let mut preds: [(usize, f32); 15] = [(0, 0.); 15];
    for iter in 0..SESSIONS {
        epsilon *= EPSILON_DECAY;
        let mut rewards_per_action: [f32; ACTION_SPACE] = [0.; ACTION_SPACE];
        let mut actions: [u16; ACTION_SPACE] = [0; ACTION_SPACE];
        let mut score: i16 = 0;
        let mut epsilon_actions: u16 = 0;
        loop {
            loss_derivative = Array1::zeros(ACTION_SPACE);
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
                epsilon_actions += 1;
                rng.random_range(0..ACTION_SPACE)
            };
            actions[choice] += 1;
            let (reward, finished) = game.step(choice, false);
            rewards_per_action[choice] = reward as f32;
            rewards[game.steps as usize - 1] = reward;
            preds[game.steps as usize - 1] = (choice, output[choice]);
            game.state().to_vec(&mut state);
            if finished {
                loss_derivative[choice] = output[choice] - reward as f32;
            } else {
                let target_output = target.forward(&state);
                let next_state_choice: usize = target_output
                    .indexed_iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, v)| i)
                    .unwrap();
                let next_state_value_estimate: f32 = target_output[next_state_choice];
                loss_derivative[choice] =
                    output[choice] - (reward as f32 + GAMMA * next_state_value_estimate);
                target_preds[game.steps as usize - 1] =
                    (next_state_choice, next_state_value_estimate);
            }
            errors[game.steps as usize - 1] = (choice, loss_derivative[choice]);
            if iter > 300 && score == -114 {
                println!("It's debugging time");
                println!("It's debugging time");
            }
            agent.backprop(&acted_upon_state, &mut loss_derivative);
            if (game.steps + residual) % BATCH_SIZE == 0 {
                agent.apply_gradients(LEARNING_RATE, BATCH_SIZE);
            }
            score += reward;
            if finished {
                residual = game.steps % BATCH_SIZE;
                // display_progress(iter);
                for i in 0..OBSERVATION_SPACE {
                    rewards_per_action[i] = if actions[i] == 0 {
                        0.
                    } else {
                        rewards_per_action[i] / actions[i] as f32
                    };
                }
                println!(
                    "Score: {}\tAction Count: {:?}\tRewards: {:?}\tOutput: {:?}",
                    score, actions, rewards_per_action, output
                );
                if iter % 2 == 0 {
                    target = agent.clone();
                }
                game.reset();
                if iter > 300
                    && epsilon_actions == 0
                    && prev_score > -15
                    && score < -15
                    && score < prev_score
                {
                    panic!(
                        "JUMPED FROM OPTIMAL:\tPrev Score: {} New Score: {}\nPrev Rewards:{:?}\nPrev Errors:{:?}\nPrev Preds:{:?}\nPrev Target Preds:{:?}",
                        prev_score, score, prev_rewards, prev_errors, prev_preds, prev_target_preds
                    );
                }
                prev_score = score;
                prev_rewards = rewards;
                prev_errors = errors;
                prev_preds = preds;
                prev_target_preds = target_preds;
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
