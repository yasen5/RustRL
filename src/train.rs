use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::distr::Uniform;

const SESSIONS: u16 = 500;
const ITER_DISPLAY_PRECISION: u16 = 20;
const LOG_INTERVAL: u16 = SESSIONS / ITER_DISPLAY_PRECISION;

pub fn fake_train(agent: &mut crate::model::Model) {
    let state: Array1<f32> = Array1::random(5, Uniform::new(-1.0, 1.0).unwrap());
    let mut loss_derivative;
    for iter in 0..SESSIONS {
        for i in 0..100 {
            loss_derivative = Array1::zeros(5);
            let (choice, reward_prediction): (usize, f32) = agent
                .forward(&state)
                .indexed_iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, v)| (i, *v))
                .unwrap();
            loss_derivative[choice] = reward_prediction - 5.;
            agent.backprop(&state, &mut loss_derivative);
            if i == 99 {
                println!("Reward prediction: {}", reward_prediction);
            }
        }
        agent.apply_gradients();
        display_progress(iter);
    }
}

pub fn train(game: &mut crate::game::Game, agent: &mut crate::model::Model) {
    let mut target: crate::model::Model = agent.clone();
    let mut state: Array1<f32> = Array1::zeros(5);
    let mut loss_derivative;
    let mut actions: [u16; 4] = [0; 4];
    for iter in 0..SESSIONS {
        game.reset();
        let mut score: i16 = 0;
        loop {
            loss_derivative = Array1::zeros(4);
            game.state().to_vec(&mut state);
            let current_state = state.clone();
            let output = agent.forward(&state);
            let (choice, reward_prediction): (usize, f32) = output
                .indexed_iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, v)| (i, *v))
                .unwrap();
            actions[choice] += 1;
            let (reward, finished) = game.step(choice);
            game.state().to_vec(&mut state);
            let target_output = target.forward(&state);
            let next_state_value_estimate: f32 = target_output
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .map(|v| *v).unwrap();
            if target_output.indexed_iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| i)
            .unwrap() == 3 {
                panic!("Target chose 3 like an idiot, output: {:?}", output);
            }
            if choice == 3 {
                panic!("Agent chose 3 like an idiot, output: {:?}", output);
            }
            loss_derivative[choice] = reward_prediction - reward as f32 + next_state_value_estimate;
            agent.backprop(&current_state, &mut loss_derivative);
            score += reward;
            if finished {
                agent.apply_gradients();
                display_progress(iter);
                println!("Score: {}\t Action Count: {:?}", score, actions);
                break;
            }
        }
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
