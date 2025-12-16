extern crate uom;

use std::f32::consts::PI;

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

const ENV_BOX_WIDTH: u16 = 100;
const ENV_BOX_HEIGHT: u16 = 100;

// TODO should this contain the engine states?
struct Rocket {
    x: Length,
    y: Length,
    vx: Velocity,
    vy: Velocity,
    tilt: Angle, // deviation from 90 degrees vertical
    angular_velocity: AngularVelocity,
    width: Length,
    height: Length,
    mass: Mass,
    lander_angle: Angle,
    lander_length: Length,
    engine_strength: Force,
    moment_of_inertia: MomentOfInertia,
}

impl Rocket {
    fn to_vec(&self) -> Vec<f32> {
        return vec![self.x.value, self.y.value, self.vx.value, self.vy.value];
    }
}

impl Rocket {
    fn new(rand_x: bool, rand_y: bool, xvel: f32, yvel: f32) -> Self {
        let width = Length::new::<meter>((ENV_BOX_WIDTH / 20).to_f32().unwrap());
        let height = Length::new::<meter>((ENV_BOX_HEIGHT / 20).to_f32().unwrap());
        let mass = Mass::new::<kilogram>(50.);
        Self {
            x: if rand_x {
                Length::new::<meter>(rand::random_range(0..ENV_BOX_WIDTH).to_f32().unwrap())
            } else {
                Length::new::<meter>(ENV_BOX_WIDTH.to_f32().unwrap() / 2.)
            },
            y: if rand_y {
                Length::new::<meter>(rand::random_range(0..ENV_BOX_HEIGHT).into())
            } else {
                Length::new::<meter>(ENV_BOX_HEIGHT.to_f32().unwrap() / 2.)
            },
            vx: Velocity::new::<meter_per_second>(xvel),
            vy: Velocity::new::<meter_per_second>(yvel),
            tilt: Angle::new::<radian>(0.),
            angular_velocity: AngularVelocity::new::<radian_per_second>(0.),
            width: width,
            height: height,
            lander_angle: Angle::new::<radian>(PI / 3.),
            lander_length: Length::new::<meter>((ENV_BOX_HEIGHT / 40).to_f32().unwrap()),
            engine_strength: Force::new::<newton>(2.),
            mass: mass,
            moment_of_inertia: mass * (width * width + height * height) / 12.0,
        }
    }
}

pub struct Game {
    state: Rocket,
    steps: u16,
}

// 0: right engine
// 1: left engine
// 2: down engine
// 3: no engines firing
impl Game {
    pub fn step(&mut self, choice: u8) -> bool {
        let DT: Time = Time::new::<second>(0.02);
        let ENGINE_ACCEL = self.state.engine_strength / self.state.mass;
        let GRAVITY: Acceleration = Acceleration::new::<meter_per_second_squared>(9.81);
        match choice {
            0 => {
                self.state.vx -= ENGINE_ACCEL * DT * self.state.tilt.cos();
                self.state.vy -= ENGINE_ACCEL * DT * self.state.tilt.sin();
                Ok(())
            }
            1 => {
                self.state.vx += ENGINE_ACCEL * DT * self.state.tilt.cos();
                self.state.vy += ENGINE_ACCEL * DT * self.state.tilt.sin();
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
    }
}
