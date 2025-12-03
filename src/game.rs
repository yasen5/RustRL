use crate::model;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand;

const TRAIN_BOX_WIDTH: u16 = 100;
const TRAIN_BOX_HEIGHT: u16 = 100;
const SESSIONS: u16 = 40;
const ITER_DISPLAY_PRECISION: u16 = 20;
const LOG_INTERVAL: u16 = SESSIONS / ITER_DISPLAY_PRECISION;

// TODO should this contain the engine states?
struct RocketState {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

impl RocketState {
    fn to_vec(&self) -> Vec<f32> {
        return vec![self.x, self.y, self.vx, self.vy];
    }
}

impl RocketState {
    fn new(rand_x: bool, rand_y: bool, xvel: f32, yvel: f32) -> Self {
        Self {
            x: if rand_x {
                rand::random_range(0..TRAIN_BOX_WIDTH).into()
            } else {
                TRAIN_BOX_WIDTH.to_f32().unwrap() / 2.
            },
            y: if rand_y {
                rand::random_range(0..TRAIN_BOX_HEIGHT).into()
            } else {
                TRAIN_BOX_HEIGHT.to_f32().unwrap() / 2.
            },
            vx: xvel,
            vy: yvel,
        }
    }
}

pub fn display_progress(iter: u16) {
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
