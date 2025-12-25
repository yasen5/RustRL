use lazy_static::lazy_static;
use macroquad::{color::Color, math::Vec2, shapes::{DrawRectangleParams, draw_circle, draw_line, draw_rectangle_ex}};
use uom::si::{f32::{Angle, Length}, length::meter};

use crate::game::Pos;

lazy_static! {
    pub static ref ENV_BOX_WIDTH: Length = Length::new::<meter>(100.);
    pub static ref ENV_BOX_HEIGHT: Length = Length::new::<meter>(100.);
    pub static ref GRAPHICS_SCALAR: f32 = 800. / ENV_BOX_HEIGHT.value;
}

#[inline]
pub fn transform_with_units(vector: &mut Pos, angle: Angle) {
    let (sin, cos) = angle.value.sin_cos();
    *vector = Pos {
        x: vector.x * cos - vector.y * sin,
        y: vector.x * sin + vector.y * cos,
    }
}

#[inline]
pub fn adjusted_draw_circle(x: Length, y: Length, radius: Length, color: Color) {
    draw_circle(
        *GRAPHICS_SCALAR * x.value,
        *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - y).value,
        *GRAPHICS_SCALAR * radius.value,
        color,
    );
}

#[inline]
pub fn adjusted_draw_rectangle_ex(
    x: Length,
    y: Length,
    w: Length,
    h: Length,
    rotation: Angle,
    color: Color,
) {
    draw_rectangle_ex(
        *GRAPHICS_SCALAR * x.value,
        *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - y).value,
        *GRAPHICS_SCALAR * w.value,
        *GRAPHICS_SCALAR * h.value,
        DrawRectangleParams {
            rotation: -rotation.value,
            offset: Vec2 { x: 0.5, y: 0.5 },
            color: color,
        },
    );
}

#[inline]
pub fn adjusted_draw_line(x1: Length, y1: Length, x2: Length, y2: Length, color: Color) {
    draw_line(
        *GRAPHICS_SCALAR * x1.value,
        *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - y1).value,
        *GRAPHICS_SCALAR * x2.value,
        *GRAPHICS_SCALAR * (*ENV_BOX_HEIGHT - y2).value,
        2.,
        color,
    );
}
