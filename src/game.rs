use std::f32::consts::PI;

use crate::graphics;
use crate::graphics::ENV_BOX_HEIGHT;
use crate::graphics::ENV_BOX_WIDTH;
use lazy_static::lazy_static;
use macroquad::color::{BLACK, BLUE, Color, GRAY, PURPLE, WHITE};
use macroquad::shapes::draw_rectangle;
use macroquad::{
    input::{KeyCode, is_key_pressed},
    text::draw_text,
    window::{clear_background, next_frame},
};
use ndarray::Array1;
use rand;
use std::{thread, time::Duration};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use uom::ConstZero;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f32::*;
use uom::si::force::newton;
use uom::si::length::meter;
use uom::si::mass::kilogram;
use uom::si::time::{millisecond, second};
use uom::si::velocity::meter_per_second;

lazy_static! {
    static ref PARTICLE_SPEED: Velocity =
        Velocity::new::<meter_per_second>(ENV_BOX_HEIGHT.value / 5.);
    static ref PARTICLE_RADIUS: Length = *ENV_BOX_HEIGHT / 100.;
    static ref PARTICLE_LIFTIME: Time = Time::new::<second>(1.);
    pub static ref MAX_STEPS: u16 = 2;
    static ref MIN_HEIGHT: Length = *ENV_BOX_HEIGHT / 5.;
    static ref MAX_ANGULAR_VEL: AngularVelocity =
        AngularVelocity::new::<radian_per_second>(PI / 4.);
    static ref MAX_VEL: Velocity = Velocity::new::<meter_per_second>(5.);
    static ref DT: Time = Time::new::<second>(1.0);
    static ref GRAVITY: Acceleration = Acceleration::new::<meter_per_second_squared>(9.81);
}

#[derive(Debug, EnumIter)]
enum Engine {
    LEFT,
    RIGHT,
    DOWN,
}

pub struct Pos {
    pub x: Length,
    pub y: Length,
}

#[derive(Clone)]
struct JetParticle {
    x: Length,
    y: Length,
    vx: Velocity,
    vy: Velocity,
    life: Time,
}

impl JetParticle {
    fn new() -> Self {
        Self {
            x: Length::new::<meter>(0.),
            y: Length::new::<meter>(0.),
            vx: Velocity::new::<meter_per_second>(0.),
            vy: Velocity::new::<meter_per_second>(0.),
            life: Time::new::<second>(0.),
        }
    }

    fn activate(&mut self, x: Length, y: Length, direction: Angle) {
        self.x = x;
        self.y = y;
        self.life = Time::new::<second>(0.);
        self.vx = *PARTICLE_SPEED * direction.cos();
        self.vy = *PARTICLE_SPEED * direction.sin();
    }

    fn update(&mut self) {
        self.x += self.vx * (*DT);
        self.y += self.vy * (*DT);
        self.life += *DT;
    }
}

pub struct Rocket {
    pos: Pos,
    vx: Velocity,
    vy: Velocity,
    tilt: Angle, // counterclockwise from x-axis
    angular_velocity: AngularVelocity,
    width: Length,
    height: Length,
    lander_angle: Angle,
    lander_length: Length,
    engine_dim: Length,
    jet_particles: [JetParticle; 15],
    particle_index: usize,
    translational_engine_accel: Acceleration,
    angular_engine_accel: AngularAcceleration,
}

const RAND_X: bool = false;
const RAND_Y: bool = false;
const START_XVEL: f32 = 0.;
const START_YVEL: f32 = 0.;

impl Rocket {
    pub fn new() -> Self {
        let mass = Mass::new::<kilogram>(50.);
        let width = *ENV_BOX_WIDTH / 10.0;
        let height = *ENV_BOX_HEIGHT / 20.0;
        let engine_strength = Force::new::<newton>(450.5);
        let engine_accel = engine_strength / mass;
        let horizontal_moi: MomentOfInertia = mass * height * height / 12.;
        let side_engine_torque: Torque = (engine_strength / 2. * height / 2.0).into(); // side engines weaker
        let side_accel: AngularAcceleration = (side_engine_torque / horizontal_moi / 30.).into();
        Self {
            pos: Pos {
                x: if RAND_X {
                    Length::new::<meter>(rand::random_range(0.0..ENV_BOX_WIDTH.value))
                } else {
                    *ENV_BOX_WIDTH
                },
                y: if RAND_Y {
                    Length::new::<meter>(rand::random_range(0.0..ENV_BOX_HEIGHT.value))
                } else {
                    *ENV_BOX_HEIGHT / 2.
                },
            },
            vx: Velocity::new::<meter_per_second>(START_XVEL),
            vy: Velocity::new::<meter_per_second>(START_YVEL),
            tilt: Angle::new::<radian>(0.),
            angular_velocity: AngularVelocity::new::<radian_per_second>(0.),
            width: width,
            height: height,
            lander_angle: Angle::new::<radian>(-PI / 3.),
            lander_length: height,
            engine_dim: width / 4.,
            jet_particles: core::array::from_fn(|_| JetParticle::new()),
            particle_index: 0,
            translational_engine_accel: engine_accel,
            angular_engine_accel: side_accel,
        }
    }

    pub fn to_vec(&self, output: &mut Array1<f32>) {
        output[0] = self.pos.x.value / ENV_BOX_WIDTH.value;
        // output[1] = self.pos.y.value / ENV_BOX_HEIGHT.value;
        output[1] = self.vx.value / GRAVITY.value; // REMEMBER TO CHANGE THIS BACK TO 2
        // output[3] = self.vy.value / GRAVITY.value;
        // output[4] = self.angular_velocity.value;
        // output[5] = self.tilt.cos().value;
        // output[6] = self.tilt.sin().value;
    }

    fn leg_pos(&self, left: bool, start: bool) -> Pos {
        let inversion = if left { -1. } else { 1. };
        let mut leg_pos = Pos {
            x: inversion
                * (if start {
                    self.width / 2.
                } else {
                    self.width / 2. + self.lander_length * self.lander_angle.cos()
                }),
            y: -(if start {
                self.height / 2.
            } else {
                self.height / 2. + self.lander_length * self.lander_angle.cos()
            }),
        };
        graphics::transform_with_units(&mut leg_pos, self.tilt);
        leg_pos.x += self.pos.x;
        leg_pos.y += self.pos.y;
        leg_pos
    }

    fn engine_pos(&self, engine: Engine) -> Pos {
        let mut engine_center_offset: Pos;
        match engine {
            Engine::RIGHT => {
                engine_center_offset = Pos {
                    x: (self.width / 2. + self.engine_dim / 2.),
                    y: (self.height / 2.),
                };
            }
            Engine::LEFT => {
                engine_center_offset = Pos {
                    x: -(self.width / 2. + self.engine_dim / 2.),
                    y: (self.height / 2.),
                };
            }
            Engine::DOWN => {
                engine_center_offset = Pos {
                    x: Length::ZERO,
                    y: -(self.height / 2.),
                }
            }
        }
        graphics::transform_with_units(&mut engine_center_offset, self.tilt);
        engine_center_offset
    }

    fn fire_engine(&mut self, engine: Engine) {
        match engine {
            Engine::RIGHT => {
                self.vx -= Velocity::new::<meter_per_second>(1.); 
            }
            Engine::LEFT => {
                self.vx += Velocity::new::<meter_per_second>(1.); 
            }
            Engine::DOWN => {
                panic!("Shouldn't fire down");
            }
        }
    }

    fn update(&mut self) {
        self.pos.x += self.vx * (*DT);
    }

    fn draw_engine(&self, engine: Engine) {
        let engine_center_offset = self.engine_pos(engine);
        graphics::adjusted_draw_rectangle_ex(
            engine_center_offset.x + self.pos.x,
            engine_center_offset.y + self.pos.y,
            self.engine_dim,
            self.engine_dim,
            self.tilt,
            GRAY,
        );
    }

    fn draw(&self) -> () {
        graphics::adjusted_draw_rectangle_ex(
            self.pos.x,
            self.pos.y,
            self.width,
            self.height,
            self.tilt,
            PURPLE,
        );
        for engine_type in Engine::iter() {
            self.draw_engine(engine_type);
        }
        let left_start: Pos = self.leg_pos(true, true);
        let left_end: Pos = self.leg_pos(true, false);
        let right_start: Pos = self.leg_pos(false, true);
        let right_end: Pos = self.leg_pos(false, false);
        graphics::adjusted_draw_line(left_start.x, left_start.y, left_end.x, left_end.y, BLUE);
        graphics::adjusted_draw_line(right_start.x, right_start.y, right_end.x, right_end.y, BLUE);
        for particle in &self.jet_particles {
            graphics::adjusted_draw_circle(
                particle.x,
                particle.y,
                *PARTICLE_RADIUS,
                Color::new(
                    1.,
                    1.,
                    1.,
                    ((*PARTICLE_LIFTIME - particle.life) / *PARTICLE_LIFTIME).value,
                ),
            );
        }
    }
}

pub struct Game {
    pub state: Rocket,
    pub action_space: usize,
    pub observation_space: usize,
    pub steps: u16,
}

impl Game {
    pub fn new() -> Self {
        Self {
            state: Rocket::new(),
            action_space: 6,
            observation_space: 5,
            steps: 0,
        }
    }
}

// 0: right engine
// 1: left engine
// 2: down engine
// 3: no engines firing
impl Game {
    #[allow(non_snake_case)]
    pub fn step(&mut self, choice: usize, verbose: bool) -> (f32, bool) {
        self.steps += 1;
        let mut reward: f32 = 0.;
        match choice {
            0 => Ok(self.state.fire_engine(Engine::RIGHT)),
            1 => Ok(self.state.fire_engine(Engine::LEFT)),
            2 => Ok(self.state.fire_engine(Engine::DOWN)),
            3 => Ok(()), // saving fuel,
            // 4 => {
            //     self.state.fire_engine(Engine::RIGHT);
            //     self.state.fire_engine(Engine::DOWN);
            //     Ok(())
            // }
            // 5 => {
            //     self.state.fire_engine(Engine::LEFT);
            //     self.state.fire_engine(Engine::DOWN);
            //     Ok(())
            // }
            _ => Err(()),
        }
        .unwrap();
        let mut finished = false;
        self.state.update();
        reward -= (self.state.pos.x - *ENV_BOX_WIDTH / 2.).abs().value / 1.;
        if self.steps >= *MAX_STEPS {
            finished = true;
        }
        reward *= 2.;
        return (reward, finished);
    }

    pub fn reset(&mut self) {
        self.steps = 0;
        self.state = Rocket::new();
    }

    pub fn draw(&self) {
        clear_background(BLACK);
        draw_rectangle(
            0.,
            *graphics::GRAPHICS_SCALAR * (ENV_BOX_HEIGHT.value * 4. / 5.),
            *graphics::GRAPHICS_SCALAR * ENV_BOX_WIDTH.value,
            *graphics::GRAPHICS_SCALAR * ENV_BOX_HEIGHT.value / 5.,
            WHITE,
        );
        self.state.draw();
    }

    #[inline]
    pub fn state(&self) -> &Rocket {
        &self.state
    }
}

pub async fn run_game(mut choose: impl FnMut() -> usize, verbose: bool) {
    let mut new_game = Game::new();
    let mut score: f32 = 0.;
    new_game.draw();
    thread::sleep(Duration::from_millis(DT.get::<millisecond>() as u64));
    loop {
        let choice: usize = choose();
        println!("Chose: {}", choice);
        // let choice = if new_game.steps <= 13 { 0 } else { 1 };

        new_game.draw();
        thread::sleep(Duration::from_millis(DT.get::<millisecond>() as u64));
        next_frame().await;

        let (reward, finished) = new_game.step(choice, verbose);
        score += reward;

        println!("Reward: {}", reward);
        if finished {
            break;
        }
    }

    loop {
        clear_background(BLACK);
        draw_text(&format!("You scored {}", score), 100.0, 100.0, 50.0, WHITE);

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }
}
