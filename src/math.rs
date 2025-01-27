pub mod angle;
pub mod matrix;
pub mod vector;

pub use angle::{Rad, Deg};

pub type Vector2 = vector::Vector<f32, 2>;
pub type Vector3 = vector::Vector<f32, 3>;
pub type Vector4 = vector::Vector<f32, 4>;

pub type Matrix2 = matrix::Matrix<f32, 2>;
pub type Matrix3 = matrix::Matrix<f32, 3>;
pub type Matrix4 = matrix::Matrix<f32, 4>;

/// Perspective matrix that is suitable for Vulkan.
///
/// It inverts the projected y-axis and sets the depth range to 0..1
/// instead of -1..1. Mind the vertex winding order though.
pub fn perspective<F>(fovy: F, aspect: f32, near: f32, far: f32) -> Matrix4
where
    F: Into<angle::Rad<f32>>,
{
    let f = 1. / (fovy.into().0 / 2.).tan();
    Matrix4::from([
        Vector4::from([f / aspect, 0., 0., 0.]),
        Vector4::from([0., -f, 0., 0.]),
        Vector4::from([0., 0., -far / (far - near), -1.]),
        Vector4::from([0., 0., -(far * near) / (far - near), 0.]),
    ])
}
