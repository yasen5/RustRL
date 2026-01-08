use std::vec;

use ndarray::{Array1, Array2};

use crate::{game, model};

const LEARNING_RATE: f32 = 0.01;
const ITERS: usize = 5000;
const TARGET_UPDATE_FREQ: usize = 100;
const PRINT_FREQ: u16 = ITERS as u16 / 10;

pub fn test_backprop() {
    #[allow(non_snake_case)]
    let BATCH_SIZE: u16 = *game::MAX_STEPS * 16;
    let mut game: game::Game = game::Game::new();
    let mut state: Array1<f32> = Array1::zeros(2);
    let mut state_clone;
    let mut agent: model::Model = model::Model::new();
    let mut counter: u16 = 0;
    let mut actions: [u16; 2] = [0; 2];
    agent.add_layer(2, 2, false);
    let mut target: model::Model = agent.clone();
    for iter in 0..ITERS {
        game.state.to_vec(&mut state);
        let mut loss_derivative: Array1<f32> = Array1::zeros(2);
        let agent_prediction: Array1<f32> = agent.forward(&state).clone();
        let agent_choice: usize = agent_prediction
            .indexed_iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        state_clone = state.clone();
        let (reward, finished) = game.step(agent_choice);
        game.state.to_vec(&mut state);
        let target_prediction: &Array1<f32> = target.forward(&state);
        let target_choice: usize = target_prediction
            .indexed_iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        loss_derivative[agent_choice] =
            agent_prediction[agent_choice] - if finished {reward } else {reward + target_prediction[target_choice]};
        agent.backprop(&state_clone, &loss_derivative);
        actions[agent_choice] += 1;
        if (game.steps - 1) % BATCH_SIZE == 0 {
            if counter % PRINT_FREQ == 0 {
                print!("Actions:{:?} Agent Output:\t", actions);
                print_array(&agent_prediction);
                print!("Target output: ");
                print_array(&target_prediction);
                println!("\nWeights, weight gradients");
                print_mats(vec![
                    &agent.layers[0].weights,
                    &agent.layers[0].weight_gradient,
                ]);
                println!("Biases, loss");
                print_array(&agent.layers[0].biases);
                print_array(&loss_derivative);
                print!(
                    "\n====================================================================================================\n"
                );
            }
            agent.apply_gradients(LEARNING_RATE);
            counter += 1;
        }
        if iter >= ITERS - 10 {
            print!(
                "Prediction: {}\tBellman: {}\tReward: {}",
                agent_prediction[agent_choice],
                if finished {reward } else {reward + target_prediction[target_choice]},
                reward
            );
            if !finished {
                println!("\tTarget Prediction: {}", target_prediction[target_choice]);
            }
            else {
                println!();
            }
        }
        if iter % TARGET_UPDATE_FREQ == 0 {
            target = agent.clone();
        }
        if finished {
            actions.fill(0);
            game.reset();
        }
    }
}

fn print_array(array: &Array1<f32>) {
    print!("[");
    for num in array.iter() {
        print!("{}\t", num);
    }
    print!("]\t\t");
}

// must be of same dims
fn print_mats(mats: Vec<&Array2<f32>>) {
    for row in 0..mats[0].nrows() {
        for mat in mats.iter() {
            print!("[");
            for col in 0..mat.ncols() {
                print!("{}\t", mat[(row, col)]);
            }
            print!("]\t\t");
        }
        println!();
    }
    println!();
}
