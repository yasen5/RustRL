use std::f32::consts::PI;

use lazy_static::lazy_static;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand;
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
    static ref ENV_BOX_WIDTH: Length = Length::new::<meter>(100.);
    static ref ENV_BOX_HEIGHT: Length = Length::new::<meter>(100.);
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
}

impl Rocket {
    pub fn new(rand_x: bool, rand_y: bool, xvel: f32, yvel: f32) -> Self {
        let mass = Mass::new::<kilogram>(50.);
        Self {
            pos: Pos {
                x: if rand_x {
                    Length::new::<meter>(
                        rand::random_range(0..ENV_BOX_WIDTH.value.to_u32().unwrap())
                            .to_f32()
                            .unwrap(),
                    )
                } else {
                    Length::new::<meter>(ENV_BOX_WIDTH.value / 2.)
                },
                y: if rand_y {
                    Length::new::<meter>(
                        rand::random_range(0..ENV_BOX_HEIGHT.value.to_u32().unwrap())
                            .to_f32()
                            .unwrap(),
                    )
                } else {
                    Length::new::<meter>(ENV_BOX_HEIGHT.value / 2.)
                },
            },
            vx: Velocity::new::<meter_per_second>(xvel),
            vy: Velocity::new::<meter_per_second>(yvel),
            tilt: Angle::new::<radian>(0.),
            angular_velocity: AngularVelocity::new::<radian_per_second>(0.),
            width: *ENV_BOX_WIDTH / 2.0,
            height: *ENV_BOX_HEIGHT / 2.0,
            lander_angle: Angle::new::<radian>(PI / 3.),
            lander_length: *ENV_BOX_HEIGHT / 2.0,
            engine_strength: Force::new::<newton>(2.),
            mass: mass,
        }
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
        let MAX_STEPS = 100;
        let MIN_HEIGHT: Length = Length::new::<meter>(20.);
        let MAX_ANGULAR_VEL: AngularVelocity = AngularVelocity::new::<radian_per_second>(PI);
        let MAX_VEL: Velocity = Velocity::new::<meter_per_second>(PI);
        let DT: Time = Time::new::<second>(0.02);
        let ENGINE_ACCEL = self.state.engine_strength / self.state.mass;
        let GRAVITY: Acceleration = Acceleration::new::<meter_per_second_squared>(9.81);
        let VERTICAL_MOI: MomentOfInertia =
            self.state.mass * self.state.width * self.state.width / 12.;
        let HORIZONTAL_MOI: MomentOfInertia =
            self.state.mass * self.state.height * self.state.height / 12.;
        let SIDE_ENGINE_TORQUE: Torque =
            (self.state.engine_strength * self.state.height / 2.0).into();
        let SIDE_ACCEL: AngularAcceleration = (SIDE_ENGINE_TORQUE / HORIZONTAL_MOI).into();
        match choice {
            0 => {
                self.state.vx -= ENGINE_ACCEL * DT * self.state.tilt.cos();
                self.state.vy -= ENGINE_ACCEL * DT * self.state.tilt.sin();
                self.state.angular_velocity -= AngularVelocity::from(SIDE_ACCEL * DT);
                Ok(())
            }
            1 => {
                self.state.vx += ENGINE_ACCEL * DT * self.state.tilt.cos();
                self.state.vy += ENGINE_ACCEL * DT * self.state.tilt.sin();
                self.state.angular_velocity -= AngularVelocity::from(SIDE_ACCEL * DT);
                Ok(())
            }
            2 => {
                self.state.vx += ENGINE_ACCEL * DT * self.state.tilt.sin();
                self.state.vy += ENGINE_ACCEL * DT * self.state.tilt.cos();
                Ok(())
            }
            3 => Ok(()),
            _ => Err(()),
        }
        .unwrap();
        self.state.vy -= GRAVITY * DT;
        // TODO replace with more complex ground logic
        let mut score: i16 = -1;
        let mut finished = false;
        let x = self.state.pos.x + self.state.vx * DT;
        let y = self.state.pos.x + self.state.vy * DT;
        if (x - *ENV_BOX_WIDTH / 2.0).value.abs()
            < (self.state.pos.x - *ENV_BOX_WIDTH / 2.0).value.abs()
        {
            score += 1;
        }
        if self.steps > MAX_STEPS {
            score -= 5;
        }
        let left_touching = self.state.leg_pos(true).y > MIN_HEIGHT;
        let right_touching = self.state.leg_pos(true).y > MIN_HEIGHT;
        if left_touching || right_touching
            && (self.state.angular_velocity > MAX_ANGULAR_VEL
                || self.state.vx.hypot(self.state.vy) < MAX_VEL)
        {
            finished = true;
            score -= 50;
        } else if left_touching && right_touching {
            finished = true;
            score += 50;
        }
        self.state.pos = Pos { x: x, y: y };
        StepOutcome {
            score: score,
            finished: finished,
        }
    }
}
