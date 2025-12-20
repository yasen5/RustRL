use std::f32::consts::PI;

use lazy_static::lazy_static;
use macroquad::color::{BLACK, Color, GRAY, LIGHTGRAY, PURPLE, WHITE};
use macroquad::math::Vec2;
use macroquad::shapes::{DrawRectangleParams, draw_circle, draw_rectangle, draw_rectangle_ex};
use macroquad::window::clear_background;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::angle::{degree, radian};
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
    static ref MAX_VEL: Velocity = Velocity::new::<meter_per_second>(PI);
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
}

impl Rocket {
    fn to_vec(&self) -> Vec<f32> {
        return vec![
            self.pos.x.value,
            self.pos.y.value,
            self.vx.value,
            self.vy.value,
        ];
    }

    fn leg_pos(&self, left: bool) -> Pos {
        let inversion = if left { 1. } else { -1. };
        Pos {
            x: self.pos.x
                + self.width * self.tilt.cos() * inversion
                + self.lander_length * (self.tilt + self.lander_angle).cos() * inversion,
            y: self.pos.y
                - self.height * self.tilt.sin()
                - self.lander_length * (self.tilt + self.lander_angle).sin(),
        }
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
        transform(&mut engine_center_offset, -self.tilt);
        engine_center_offset
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
                rotation: self.tilt.value,
                color: GRAY,
            },
        );
    }
}

impl Rocket {
    pub fn new(rand_x: bool, rand_y: bool, xvel: f32, yvel: f32) -> Self {
        let mass = Mass::new::<kilogram>(50.);
        let width = *ENV_BOX_WIDTH / 10.0;
        let height = *ENV_BOX_HEIGHT / 20.0;
        Self {
            pos: Pos {
                x: if rand_x {
                    Length::new::<meter>(
                        rand::random_range(0..ENV_BOX_WIDTH.value.to_u32().unwrap())
                            .to_f32()
                            .unwrap(),
                    )
                } else {
                    *ENV_BOX_WIDTH / 2.
                },
                y: if rand_y {
                    Length::new::<meter>(
                        rand::random_range(0..ENV_BOX_HEIGHT.value.to_u32().unwrap())
                            .to_f32()
                            .unwrap(),
                    )
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
            lander_angle: Angle::new::<radian>(PI / 3.),
            lander_length: *ENV_BOX_HEIGHT / 2.0,
            engine_strength: Force::new::<newton>(1000.),
            mass: mass,
            engine_dim: width / 4.,
        }
    }
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

pub struct Game {
    state: Rocket,
    jet_particles: Vec<JetParticle>,
    steps: u16,
}

impl Game {
    pub fn new() -> Self {
        Self {
            state: Rocket::new(false, false, 0., 0.),
            jet_particles: vec![],
            steps: 0,
        }
    }
}

pub struct StepOutcome {
    score: i16,
    finished: bool,
}

// 0: right engine
// 1: left engine
// 2: down engine
// 3: no engines firing
impl Game {
    #[allow(non_snake_case)]
    pub fn step(&mut self, choice: u8) -> StepOutcome {
        // TODO move constants
        let ENGINE_ACCEL = self.state.engine_strength / self.state.mass;
        let VERTICAL_MOI: MomentOfInertia =
            self.state.mass * self.state.width * self.state.width / 12.;
        let HORIZONTAL_MOI: MomentOfInertia =
            self.state.mass * self.state.height * self.state.height / 12.;
        let SIDE_ENGINE_TORQUE: Torque =
            (self.state.engine_strength * self.state.height / 2.0).into();
        let SIDE_ACCEL: AngularAcceleration = (SIDE_ENGINE_TORQUE / HORIZONTAL_MOI / 30.).into();
        match choice {
            0 => {
                self.state.vx -= ENGINE_ACCEL * (*DT) * self.state.tilt.cos();
                self.state.vy -= ENGINE_ACCEL * (*DT) * self.state.tilt.sin();
                self.state.angular_velocity -= AngularVelocity::from(SIDE_ACCEL * (*DT));
                let right_engine_pos = self.state.engine_pos(Engine::RIGHT);
                self.jet_particles.push(JetParticle::new(
                    self.state.pos.x + Length::new::<meter>(right_engine_pos.x),
                    self.state.pos.y + Length::new::<meter>(right_engine_pos.y),
                    self.state.tilt,
                ));
                Ok(())
            }
            1 => {
                self.state.vx += ENGINE_ACCEL * (*DT) * self.state.tilt.cos();
                self.state.vy += ENGINE_ACCEL * (*DT) * self.state.tilt.sin();
                self.state.angular_velocity += AngularVelocity::from(SIDE_ACCEL * (*DT));
                let left_engine_pos = self.state.engine_pos(Engine::LEFT);
                self.jet_particles.push(JetParticle::new(
                    self.state.pos.x + Length::new::<meter>(left_engine_pos.x),
                    self.state.pos.y + Length::new::<meter>(left_engine_pos.y),
                    self.state.tilt - Angle::new::<radian>(PI),
                ));
                Ok(())
            }
            2 => {
                self.state.vx += ENGINE_ACCEL * 2. * (*DT) * self.state.tilt.sin();
                self.state.vy += ENGINE_ACCEL * 2. * (*DT) * self.state.tilt.cos();
                let down_engine_pos = self.state.engine_pos(Engine::DOWN);
                println!("Spawning with dir {} \tDue to tilt {}", (self.state.tilt - Angle::new::<radian>(PI / 2.)).get::<degree>(), self.state.tilt.get::<degree>());
                self.jet_particles.push(JetParticle::new(
                    self.state.pos.x + Length::new::<meter>(down_engine_pos.x),
                    self.state.pos.y + Length::new::<meter>(down_engine_pos.y),
                    self.state.tilt - Angle::new::<radian>(PI / 2.),
                ));
                Ok(())
            }
            3 => Ok(()),
            _ => Err(()),
        }
        .unwrap();
        self.state.vy -= (*GRAVITY) * (*DT);
        let mut score: i16 = -1;
        let mut finished = false;
        let x = self.state.pos.x + self.state.vx * (*DT);
        let y = self.state.pos.y + self.state.vy * (*DT);
        self.state.tilt += Angle::from(self.state.angular_velocity * (*DT));
        if (x - *ENV_BOX_WIDTH / 2.0).value.abs()
            < (self.state.pos.x - *ENV_BOX_WIDTH / 2.0).value.abs()
        {
            score += 1;
        }
        if self.steps > *MAX_STEPS {
            score -= 5;
        }
        let left_touching = self.state.leg_pos(true).y > *MIN_HEIGHT;
        let right_touching = self.state.leg_pos(true).y > *MIN_HEIGHT;
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
        self.state.pos = Pos { x: x, y: y };
        for particle in &mut self.jet_particles {
            particle.update();
        }
        self.jet_particles.retain(|p| p.life < *PARTICLE_LIFTIME);
        StepOutcome {
            score: score,
            finished: finished,
        }
    }

    pub fn draw(&self, choice: u8) {
        clear_background(BLACK);
        draw_rectangle(
            0.,
            *GRAPHICS_SCALAR * (ENV_BOX_HEIGHT.value * 4. / 5.),
            *GRAPHICS_SCALAR * ENV_BOX_WIDTH.value,
            *GRAPHICS_SCALAR * ENV_BOX_HEIGHT.value / 5.,
            WHITE,
        );
        draw_rectangle_ex(
            *GRAPHICS_SCALAR * self.state.pos.x.value,
            *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - self.state.pos.y).value,
            *GRAPHICS_SCALAR * self.state.width.value,
            *GRAPHICS_SCALAR * self.state.height.value,
            DrawRectangleParams {
                rotation: self.state.tilt.value, // radians
                offset: Vec2 { x: 0.5, y: 0.5 },
                color: PURPLE,
            },
        );
        for engine_type in Engine::iter() {
            self.state.draw_engine(engine_type);
        }
        for particle in &self.jet_particles {
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
    }
}

fn transform(vector: &mut Vec2, angle: Angle) {
    *vector = Vec2 {
        x: vector.x * angle.cos().value - vector.y * angle.sin().value,
        y: vector.x * angle.sin().value + vector.y * angle.cos().value,
    }
}
