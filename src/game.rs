use std::f32::consts::PI;

use lazy_static::lazy_static;
use macroquad::color::{BLACK, BLUE, Color, GRAY, PURPLE, WHITE};
use macroquad::math::Vec2;
use macroquad::shapes::{
    DrawRectangleParams, draw_circle, draw_line, draw_rectangle, draw_rectangle_ex,
};
use macroquad::window::clear_background;
use rand;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f32::*;
use uom::si::force::newton;
use uom::si::length::meter;
use uom::si::mass::kilogram;
use uom::si::time::second;
use uom::si::velocity::meter_per_second;

lazy_static! {
    static ref ENV_BOX_WIDTH: Length = Length::new::<meter>(500.);
    static ref ENV_BOX_HEIGHT: Length = Length::new::<meter>(500.);
    static ref GRAPHICS_SCALAR: f32 = 800. / ENV_BOX_HEIGHT.value;
    static ref PARTICLE_SPEED: Velocity = Velocity::new::<meter_per_second>(128.);
    static ref PARTICLE_RADIUS: Length = Length::new::<meter>(4.);
    static ref PARTICLE_LIFTIME: Time = Time::new::<second>(1.);
    static ref MAX_STEPS: u16 = 100;
    static ref MIN_HEIGHT: Length = *ENV_BOX_HEIGHT / 5.;
    static ref MAX_ANGULAR_VEL: AngularVelocity = AngularVelocity::new::<radian_per_second>(PI);
    static ref MAX_VEL: Velocity = Velocity::new::<meter_per_second>(0.);
    static ref DT: Time = Time::new::<second>(0.1);
    static ref GRAVITY: Acceleration = Acceleration::new::<meter_per_second_squared>(9.81);
}

#[derive(Debug, EnumIter)]
enum Engine {
    LEFT,
    RIGHT,
    DOWN,
}

pub struct Pos {
    x: Length,
    y: Length,
}

struct JetParticle {
    x: Length,
    y: Length,
    vx: Velocity,
    vy: Velocity,
    life: Time,
}

impl JetParticle {
    fn new(x: Length, y: Length, direction: Angle) -> Self {
        Self {
            x: x,
            y: y,
            vx: direction.cos() * (*PARTICLE_SPEED),
            vy: direction.sin() * (*PARTICLE_SPEED),
            life: Time::new::<second>(0.),
        }
    }

    fn update(&mut self) {
        self.x += self.vx * (*DT);
        self.y += self.vy * (*DT);
        self.life += *DT;
    }
}

// TODO should this contain the engine states?
pub struct Rocket {
    pos: Pos,
    vx: Velocity,
    vy: Velocity,
    tilt: Angle, // counterclockwise from x-axis
    angular_velocity: AngularVelocity,
    width: Length,
    height: Length,
    mass: Mass,
    lander_angle: Angle,
    lander_length: Length,
    engine_strength: Force,
    engine_dim: Length,
    jet_particles: Vec<JetParticle>,
}

impl Rocket {
    pub fn new(rand_x: bool, rand_y: bool, xvel: f32, yvel: f32) -> Self {
        let mass = Mass::new::<kilogram>(50.);
        let width = *ENV_BOX_WIDTH / 10.0;
        let height = *ENV_BOX_HEIGHT / 20.0;
        Self {
            pos: Pos {
                x: if rand_x {
                    Length::new::<meter>(rand::random_range(0.0..ENV_BOX_WIDTH.value))
                } else {
                    *ENV_BOX_WIDTH / 2.
                },
                y: if rand_y {
                    Length::new::<meter>(rand::random_range(0.0..ENV_BOX_HEIGHT.value))
                } else {
                    *ENV_BOX_HEIGHT / 2.
                },
            },
            vx: Velocity::new::<meter_per_second>(xvel),
            vy: Velocity::new::<meter_per_second>(yvel),
            tilt: Angle::new::<radian>(0.),
            angular_velocity: AngularVelocity::new::<radian_per_second>(0.),
            width: width,
            height: height,
            lander_angle: Angle::new::<radian>(-PI / 3.),
            lander_length: height,
            engine_strength: Force::new::<newton>(1000.),
            mass: mass,
            engine_dim: width / 4.,
            jet_particles: vec![],
        }
    }

    fn to_vec(&self) -> Vec<f32> {
        return vec![
            self.pos.x.value,
            self.pos.y.value,
            self.vx.value,
            self.vy.value,
        ];
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
        transform_with_units(&mut leg_pos, self.tilt);
        leg_pos.x += self.pos.x;
        leg_pos.y += self.pos.y;
        leg_pos
    }

    fn engine_pos(&self, engine: Engine) -> Vec2 {
        let mut engine_center_offset: Vec2;
        match engine {
            Engine::RIGHT => {
                engine_center_offset = Vec2 {
                    x: (self.width / 2. + self.engine_dim / 2.).value,
                    y: (self.height / 2.).value,
                };
            }
            Engine::LEFT => {
                engine_center_offset = Vec2 {
                    x: -(self.width / 2. + self.engine_dim / 2.).value,
                    y: (self.height / 2.).value,
                };
            }
            Engine::DOWN => {
                engine_center_offset = Vec2 {
                    x: 0.,
                    y: -(self.height / 2.).value,
                }
            }
        }
        transform(&mut engine_center_offset, self.tilt);
        engine_center_offset
    }

    fn fire_engine(&mut self, engine: Engine) {
        let ENGINE_ACCEL = self.engine_strength / self.mass;
        let HORIZONTAL_MOI: MomentOfInertia = self.mass * self.height * self.height / 12.;
        let SIDE_ENGINE_TORQUE: Torque = (self.engine_strength * self.height / 2.0).into();
        let SIDE_ACCEL: AngularAcceleration = (SIDE_ENGINE_TORQUE / HORIZONTAL_MOI / 30.).into();
        match engine {
            Engine::RIGHT => {
                self.vx -= ENGINE_ACCEL * (*DT) * self.tilt.cos();
                self.vy -= ENGINE_ACCEL * (*DT) * self.tilt.sin();
                self.angular_velocity += AngularVelocity::from(SIDE_ACCEL * (*DT));
                let right_engine_pos = self.engine_pos(Engine::RIGHT);
                self.jet_particles.push(JetParticle::new(
                    self.pos.x + Length::new::<meter>(right_engine_pos.x),
                    self.pos.y + Length::new::<meter>(right_engine_pos.y),
                    self.tilt,
                ));
            }
            Engine::LEFT => {
                self.vx += ENGINE_ACCEL * (*DT) * self.tilt.cos();
                self.vy += ENGINE_ACCEL * (*DT) * self.tilt.sin();
                self.angular_velocity -= AngularVelocity::from(SIDE_ACCEL * (*DT));
                let left_engine_pos = self.engine_pos(Engine::LEFT);
                self.jet_particles.push(JetParticle::new(
                    self.pos.x + Length::new::<meter>(left_engine_pos.x),
                    self.pos.y + Length::new::<meter>(left_engine_pos.y),
                    self.tilt - Angle::new::<radian>(PI),
                ));
            }
            Engine::DOWN => {
                self.vx -= ENGINE_ACCEL * 2. * (*DT) * self.tilt.sin();
                self.vy += ENGINE_ACCEL * 2. * (*DT) * self.tilt.cos();
                let down_engine_pos = self.engine_pos(Engine::DOWN);
                self.jet_particles.push(JetParticle::new(
                    self.pos.x + Length::new::<meter>(down_engine_pos.x),
                    self.pos.y + Length::new::<meter>(down_engine_pos.y),
                    self.tilt - Angle::new::<radian>(PI / 2.),
                ));
            }
        }
    }

    fn update(&mut self) {
        self.vy -= (*GRAVITY) * (*DT);
        self.pos.x += self.vx * (*DT);
        self.pos.y += self.vy * (*DT);
        self.tilt += Angle::from(self.angular_velocity * (*DT));
        for particle in &mut self.jet_particles {
            particle.update();
        }
        self.jet_particles.retain(|p| p.life < *PARTICLE_LIFTIME);
    }

    fn draw_engine(&self, engine: Engine) {
        let engine_center_offset = self.engine_pos(engine);
        draw_rectangle_ex(
            *GRAPHICS_SCALAR * (engine_center_offset.x + self.pos.x.value),
            *GRAPHICS_SCALAR * (ENV_BOX_HEIGHT.value - (engine_center_offset.y + self.pos.y.value)),
            *GRAPHICS_SCALAR * self.engine_dim.value,
            *GRAPHICS_SCALAR * self.engine_dim.value,
            DrawRectangleParams {
                offset: Vec2 { x: 0.5, y: 0.5 },
                rotation: -self.tilt.value, // inverted bc of dumb macroquad convention of positive being clockwise
                color: GRAY,
            },
        );
    }

    fn draw(&self) -> () {
        draw_rectangle_ex(
            *GRAPHICS_SCALAR * self.pos.x.value,
            *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - self.pos.y).value,
            *GRAPHICS_SCALAR * self.width.value,
            *GRAPHICS_SCALAR * self.height.value,
            DrawRectangleParams {
                rotation: -self.tilt.value, // radians, inverted bc of stupid convention difference
                offset: Vec2 { x: 0.5, y: 0.5 },
                color: PURPLE,
            },
        );
        for engine_type in Engine::iter() {
            self.draw_engine(engine_type);
        }
        let left_start: Pos = self.leg_pos(true, true);
        let left_end: Pos = self.leg_pos(true, false);
        let right_start: Pos = self.leg_pos(false, true);
        let right_end: Pos = self.leg_pos(false, false);
        draw_line(
            *GRAPHICS_SCALAR * left_start.x.value,
            *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - left_start.y).value,
            *GRAPHICS_SCALAR * left_end.x.value,
            *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - left_end.y).value,
            2.,
            BLUE,
        );
        draw_line(
            *GRAPHICS_SCALAR * right_start.x.value,
            *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - right_start.y).value,
            *GRAPHICS_SCALAR * right_end.x.value,
            *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - right_end.y).value,
            2.,
            BLUE,
        );
    }
}

pub struct Game {
    state: Rocket,
    steps: u16,
}

impl Game {
    pub fn new() -> Self {
        Self {
            state: Rocket::new(false, false, 0., 0.),
            steps: 0,
        }
    }
}

pub struct StepOutcome {
    pub score: i16,
    pub finished: bool,
}

// 0: right engine
// 1: left engine
// 2: down engine
// 3: no engines firing
impl Game {
    #[allow(non_snake_case)]
    pub fn step(&mut self, choice: u8) -> StepOutcome {
        // TODO move constants
        match choice {
            0 => Ok(self.state.fire_engine(Engine::RIGHT)),
            1 => Ok(self.state.fire_engine(Engine::LEFT)),
            2 => Ok(self.state.fire_engine(Engine::DOWN)),
            3 => Ok(()),
            _ => Err(()),
        }
        .unwrap();
        let mut score: i16 = -1;
        let mut finished = false;
        let prev_x = self.state.pos.x;
        self.state.update();
        if (self.state.pos.x - *ENV_BOX_WIDTH / 2.0).value.abs()
            < (prev_x - *ENV_BOX_WIDTH / 2.0).value.abs()
        {
            score += 1;
        }
        if self.steps > *MAX_STEPS {
            score -= 5;
            finished = true;
        }
        let left_touching = self.state.leg_pos(false, false).y < *MIN_HEIGHT;
        let right_touching = self.state.leg_pos(true, false).y < *MIN_HEIGHT;
        if left_touching
            || right_touching
                && (self.state.angular_velocity > *MAX_ANGULAR_VEL
                    || self.state.vx.hypot(self.state.vy) < *MAX_VEL)
        {
            finished = true;
            score -= 50;
        } else if left_touching && right_touching {
            finished = true;
            score += 50;
        }
        StepOutcome {
            score: score,
            finished: finished,
        }
    }

    pub fn draw(&self) {
        clear_background(BLACK);
        draw_rectangle(
            0.,
            *GRAPHICS_SCALAR * (ENV_BOX_HEIGHT.value * 4. / 5.),
            *GRAPHICS_SCALAR * ENV_BOX_WIDTH.value,
            *GRAPHICS_SCALAR * ENV_BOX_HEIGHT.value / 5.,
            WHITE,
        );
        for particle in &self.state.jet_particles {
            draw_circle(
                *GRAPHICS_SCALAR * particle.x.value,
                *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - particle.y).value,
                *GRAPHICS_SCALAR * PARTICLE_RADIUS.value,
                Color::new(
                    1.,
                    1.,
                    1.,
                    ((*PARTICLE_LIFTIME - particle.life) / *PARTICLE_LIFTIME).value,
                ),
            );
        }
        self.state.draw();
    }
}

fn transform(vector: &mut Vec2, angle: Angle) {
    *vector = Vec2 {
        x: vector.x * angle.cos().value - vector.y * angle.sin().value,
        y: vector.x * angle.sin().value + vector.y * angle.cos().value,
    }
}

fn transform_with_units(vector: &mut Pos, angle: Angle) {
    *vector = Pos {
        x: vector.x * angle.cos().value - vector.y * angle.sin().value,
        y: vector.x * angle.sin().value + vector.y * angle.cos().value,
    }
}
