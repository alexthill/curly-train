use std::ops;

/// A fixed sized vector that is generic over its type and size.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Vector<T, const N: usize> {
    array: [T; N],
}

impl<T: Copy, const N: usize> Vector<T, N> {
    /// Creates a vector filled with `value`.
    pub fn new(value: T) -> Self {
        Self { array: [value; N] }
    }

    pub fn x(&self) -> T {
        const { assert!(N > 0, "not enough dimensions") }
        self[0]
    }
    pub fn y(&self) -> T {
        const { assert!(N > 1, "not enough dimensions") }
        self[1]
    }

    pub fn z(&self) -> T {
        const { assert!(N > 2, "not enough dimensions") }
        self[2]
    }

    pub fn w(&self) -> T {
        const { assert!(N > 3, "not enough dimensions") }
        self[3]
    }
}

impl<T: ops::Mul<Output = T> + std::iter::Sum, const N: usize> Vector<T, N> {
    /// Calculates the dot product of two vectors.
    pub fn dot(self, rhs: Self) -> T {
        self.array.into_iter()
            .zip(rhs.array)
            .map(|(a, b)| a * b)
            .sum::<T>()
    }
}

impl<const N: usize> Vector<f32, N> {
    /// Calculates the euclidian magnitude of a vector.
    pub fn magnitude(&self) -> f32 {
        self.array.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// Returns a normalized vector pointing in the same direction.
    pub fn normalize(mut self) -> Self {
        let mag = self.magnitude();
        for x in self.array.iter_mut() {
            *x /= mag;
        }
        self
    }
}

impl<T> Vector<T, 3>
where
    T: Copy + ops::Mul<Output = T> + ops::Sub<Output = T>,
{
    /// Calculates the cross product of two vectors.
    pub fn cross(self, rhs: Self) -> Self {
        Self { array: [
            self.array[1] * rhs.array[2] - self.array[2] * rhs.array[1],
            self.array[2] * rhs.array[0] - self.array[0] * rhs.array[2],
            self.array[0] * rhs.array[1] - self.array[1] * rhs.array[0],
        ] }
    }
}

impl<T: ops::Neg<Output = T>, const N: usize> ops::Neg for Vector<T, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { array: self.array.map(|x| -x) }
    }
}

impl<T: ops::AddAssign, const N: usize> ops::Add for Vector<T, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: ops::AddAssign, const N: usize> ops::AddAssign for Vector<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        for (a, b) in self.array.iter_mut().zip(rhs.array) {
            *a += b;
        }
    }
}

impl<T: ops::SubAssign, const N: usize> ops::Sub for Vector<T, N> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T: ops::SubAssign, const N: usize> ops::SubAssign for Vector<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for (a, b) in self.array.iter_mut().zip(rhs.array) {
            *a -= b;
        }
    }
}

impl<T: Copy + ops::Div<Output = T>, const N: usize> ops::Div<T> for Vector<T, N> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        Self { array: self.array.map(|x| x / rhs) }
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
    fn from(val: Vector<T, N>) -> Self {
        val.array
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(array: [T; N]) -> Self {
        Self { array }
    }
}

impl<T, const N: usize> ops::Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.array[idx]
    }
}

impl<T, const N: usize> ops::IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.array[idx]
    }
}

impl<T: Default + Copy, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Self { array: [T::default(); N] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Vector::from([1, 2]);
        let array: [_; 2] = a.into();
        assert_eq!([1, 2], array);
    }

    #[test]
    fn index() {
        let mut a = Vector::from([1, 2]);
        assert_eq!(a[0], 1);
        assert_eq!(a[1], 2);
        a[0] = 42;
        assert_eq!(a[0], 42);
    }

    #[test]
    fn neg() {
        let a = Vector::from([1, 2]);
        assert_eq!(-a, [-1, -2].into());
    }

    #[test]
    fn add() {
        let a = Vector::from([1, 2]);
        let b = Vector::from([3, 4]);
        assert_eq!(a + b, [4, 6].into());
    }

    #[test]
    fn sub() {
        let a = Vector::from([1, 2]);
        let b = Vector::from([3, 4]);
        assert_eq!(a - b, [-2, -2].into());
    }

    #[test]
    fn dot() {
        let a = Vector::from([1, 2, 3]);
        let b = Vector::from([3, 4, 5]);
        assert_eq!(a.dot(b), 26);
    }

    #[test]
    fn cross() {
        let a = Vector::from([1, 2, 3]);
        let b = Vector::from([3, 4, 5]);
        assert_eq!(a.cross(b), [-2, 4, -2].into());
    }

    #[test]
    fn magnitude_and_norm() {
        let v = Vector::from([3., 4.]);
        assert_eq!(v.magnitude(), 5.);
        let v = v.normalize();
        assert_eq!(v.magnitude(), 1.);
    }
}
