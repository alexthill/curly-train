#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rad<T>(pub T);

impl From<Deg<f32>> for Rad<f32> {
    fn from(value: Deg<f32>) -> Self {
        Rad(value.0 / 180. * std::f32::consts::PI)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Deg<T>(pub T);

impl From<Rad<f32>> for Deg<f32> {
    fn from(value: Rad<f32>) -> Self {
        Deg(value.0 * 180. / std::f32::consts::PI)
    }
}
