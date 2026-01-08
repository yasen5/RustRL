use std::vec;

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;

use crate::{game, train};

const SESSIONS: u16 = 20;
const ITER_DISPLAY_PRECISION: u16 = 20;
const LOG_INTERVAL: u16 = SESSIONS / ITER_DISPLAY_PRECISION;
const GAMMA: f32 = 0.99;
const LEARNING_RATE: f32 = 0.001;
const BATCH_SIZE: usize = 4;
const SAMPLING_FREQUENCY: usize = 5;
const TARGET_UPDATE_FREQUENCY: usize = 3;
const EPSILON_DECAY: f32 = 0.95;
pub const SEED: u64 = 42;

#[derive(Clone)]
struct Experience {
    state: Array1<f32>,
    action: usize,
    reward: f32,
    next_state: Array1<f32>,
}

struct ReplayBuffer {
    experience_replay: Vec<Experience>,
    rng: rand::rngs::StdRng,
    sample_distr: rand_distr::Normal<f32>,
}

impl ReplayBuffer {
    pub fn new() -> Self {
        Self {
            experience_replay: vec![],
            rng: rand::rngs::StdRng::seed_from_u64(train::SEED),
            sample_distr: rand_distr::Normal::new(0., 0.).unwrap(),
        }
    }

    pub fn push_experience(&mut self, experience: Experience) {
        self.experience_replay.push(experience);
    }

    pub fn sample(&mut self) -> [&Experience; BATCH_SIZE] {
        self.sample_distr =
            rand_distr::Normal::new(0., self.experience_replay.len() as f32 / 3.).unwrap();
        let experiences: [&Experience; BATCH_SIZE] = (0..BATCH_SIZE)
            .map(|_| {
                &self.experience_replay[self.sample_distr.sample(&mut self.rng).abs() as usize]
            })
            .collect::<Vec<&Experience>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to collect experiences into an array"));
        return experiences;
    }
}

pub async fn train(game: &mut crate::game::Game, agent: &mut crate::model::Model) {
    let mut replay_buffer: ReplayBuffer = ReplayBuffer::new();
    let mut rng: rand::rngs::StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut acted_upon_state: Array1<f32>;
    let mut target: crate::model::Model = agent.clone();
    let mut state: Array1<f32> = Array1::zeros(game.observation_space);
    let mut epsilon: f32 = 1.0;
    let mut sample_progress: usize = 0;
    let mut loss_derivative: Array1<f32>;
    for iter in 0..SESSIONS {
        epsilon *= EPSILON_DECAY;
        let mut score: f32 = 0.;
        loop {
            loss_derivative = Array1::zeros(game.action_space);
            game.state().to_vec(&mut state);
            acted_upon_state = state.clone();
            let agent_prediction = agent.forward(&state);
            let choice: usize = if rng.random::<f32>() > epsilon {
                agent_prediction
                    .indexed_iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap()
            } else {
                rng.random_range(0..game.action_space)
            };
            let (reward, finished) = game.step(choice);
            game.state().to_vec(&mut state);
            replay_buffer.push_experience(Experience {
                state: acted_upon_state,
                action: choice,
                reward: reward,
                next_state: state.clone(),
            });
            sample_progress += 1;
            if sample_progress % SAMPLING_FREQUENCY == 0 {
                for experience in replay_buffer.sample() {
                    let agent_reward_prediction: f32 =
                        agent.forward(&experience.state)[experience.action];
                    let next_state_reward_prediction = target
                        .forward(&experience.next_state)
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();
                    loss_derivative[experience.action] = agent_reward_prediction
                        - GAMMA * (experience.reward + next_state_reward_prediction);
                    agent.backprop(&experience.state, &loss_derivative);
                }
                agent.apply_gradients(LEARNING_RATE);
                if sample_progress % (SAMPLING_FREQUENCY * TARGET_UPDATE_FREQUENCY) == 0 {
                    target = agent.clone();
                }
            }
            score += reward;
            if finished {
                game.reset();
                println!("Scored: {}", score);
                break;
            }
        }
        display_progress(iter);
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
