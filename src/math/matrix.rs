use super::angle::Rad;
use super::vector::Vector;
use std::ops;

/// A column based matrix type that is generic over its type and size.
/// Where `M` is the number of columns and `N` the number of rows.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Matrix<T, const M: usize, const N: usize = M> {
    cols: [Vector<T, N>; M],
}

impl<T: Default + Copy, const M: usize> Matrix<T, M, M> {
    /// Creates a diagonal matrix with `value` on its main diagonal and zeroes elsewhere.
    pub fn diag(value: T) -> Self {
        let mut out = Self::default();
        for i in 0..M {
            out.cols[i][i] = value;
        }
        out
    }

    /// Returns a transposed matrix.
    pub fn transpose_sqr(mut self) -> Self {
        for i in 1..M {
            for j in 0..i/2+1 {
                let tmp = self[i][j];
                self[i][j] = self[j][i];
                self[j][i] = tmp;
            }
        }
        self
    }
}

impl<T: Default + Copy + From<bool>, const M: usize> Matrix<T, M, M> {
    /// Creates a unit matrix where the unit is derived from `true`.
    pub fn unit() -> Self {
        Self::diag(true.into())
    }
}

impl<T: Default + Copy + From<bool>, const M: usize> Matrix<T, M> {
    /// Creates a translation matrix from a translation vector.
    /// The dimension of the vector must be one less than the dimension of the matrix.
    ///
    /// #Examples
    ///
    /// ```
    /// use scop_lib::math::matrix::Matrix;
    ///
    /// let _ = Matrix::<_, 3, 3>::from_translation([1, 2].into());
    /// ```
    ///
    /// ```compile_fail
    /// use scop_lib::math::matrix::Matrix;
    ///
    /// let _ = Matrix::<_, 2, 2>::from_translation([1, 2].into());
    /// ```
    pub fn from_translation<const P: usize>(vec: Vector<T, P>) -> Self {
        const { assert!(P + 1 == M, "bad vector dimension") };

        let mut out = Self::unit();
        for i in 0..P {
            out[M - 1][i] = vec[i];
        }
        out
    }

    /// Creates a uniform scaling matrix.
    pub fn from_scale(scale: T) -> Self {
        let mut out = Self::diag(scale);
        out[M - 1][M - 1] = true.into();
        out
    }
}

impl Matrix<f32, 4> {
    /// Creates a transformation matrix that will cause a vector to point at
    /// `dir`, using `up` for orientation.
    pub fn look_to_rh(eye: Vector<f32, 3>, dir: Vector<f32, 3>, up: Vector<f32, 3>) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self::from([
            [s[0], u[0], -f[0], 0.],
            [s[1], u[1], -f[1], 0.],
            [s[2], u[2], -f[2], 0.],
            [-eye.dot(s), -eye.dot(u), eye.dot(f), 1.],
        ])
    }

    /// Create a transformation matrix that will cause a vector to point at
    /// `center`, using `up` for orientation.
    pub fn look_at_rh(eye: Vector<f32, 3>, center: Vector<f32, 3>, up: Vector<f32, 3>) -> Self {
        Self::look_to_rh(eye, center - eye, up)
    }

    /// Creates a rotation matrix around `x` axis.
    pub fn from_angle_x<A: Into<Rad<f32>>>(angle: A) -> Self {
        let (s, c) = angle.into().0.sin_cos();
        Self::from([
            [1., 0., 0., 0.],
            [0.,  c,  s, 0.],
            [0., -s,  c, 0.],
            [0., 0., 0., 1.],
        ])
    }

    /// Creates a rotation matrix around `y` axis.
    pub fn from_angle_y<A: Into<Rad<f32>>>(angle: A) -> Self {
        let (s, c) = angle.into().0.sin_cos();
        Self::from([
            [ c, 0., -s, 0.],
            [0., 1., 0., 0.],
            [ s, 0.,  c, 0.],
            [0., 0., 0., 1.],
        ])
    }

    /// Creates a rotation matrix around `z` axis.
    pub fn from_angle_z<A: Into<Rad<f32>>>(angle: A) -> Self {
        let (s, c) = angle.into().0.sin_cos();
        Self::from([
            [ c,  s, 0., 0.],
            [-s,  c, 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    }
}

impl<T: ops::AddAssign, const M: usize, const N: usize> ops::Add for Matrix<T, M, N> {
    type Output = Matrix<T, M, N>;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (a, b) in self.cols.iter_mut().zip(rhs.cols) {
            *a += b;
        }
        self
    }
}

impl<T: ops::SubAssign, const M: usize, const N: usize> ops::Sub for Matrix<T, M, N> {
    type Output = Matrix<T, M, N>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for (a, b) in self.cols.iter_mut().zip(rhs.cols) {
            *a -= b;
        }
        self
    }
}

impl<T, const M: usize, const N: usize, const P: usize> ops::Mul<Matrix<T, M, N>> for Matrix<T, N, P>
where
    T: Default + Copy + ops::AddAssign + ops::Mul<Output = T>,
{
    type Output = Matrix<T, M, P>;

    fn mul(self, rhs: Matrix<T, M, N>) -> Self::Output {
        let mut out = Self::Output::default();
        for i in 0..M {
            for j in 0..P {
                for k in 0..N {
                    out.cols[i][j] += self.cols[k][j] * rhs.cols[i][k];
                }
            }
        }
        out
    }
}

impl<T, const M: usize, const N: usize> From<Matrix<T, M, N>> for [Vector<T, N>; M] {
    fn from(val: Matrix<T, M, N>) -> Self {
        val.cols
    }
}

impl<T, const M: usize, const N: usize> From<Matrix<T, M, N>> for [[T; N]; M] {
    fn from(val: Matrix<T, M, N>) -> Self {
        val.cols.map(|col| col.into())
    }
}

impl<T, const M: usize, const N: usize> From<[Vector<T, N>; M]> for Matrix<T, M, N> {
    fn from(cols: [Vector<T, N>; M]) -> Self {
        Self { cols }
    }
}

impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N> {
    fn from(cols: [[T; N]; M]) -> Self {
        Self { cols: cols.map(|col| col.into()) }
    }
}

impl<T, const M: usize, const N: usize> ops::Index<usize> for Matrix<T, M, N> {
    type Output = Vector<T, N>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.cols[idx]
    }
}

impl<T, const M: usize, const N: usize> ops::IndexMut<usize> for Matrix<T, M, N> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.cols[idx]
    }
}

impl<T: Default + Copy, const M: usize, const N: usize> Default for Matrix<T, M, N> {
    fn default() -> Self {
        Self { cols: [Vector::<T, N>::default(); M] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a: Matrix<i32, 2> = Matrix::from([
            Vector::from([1, 2]),
            Vector::from([3, 4]),
        ]);
        let b: Matrix<i32, 2> = Matrix::from([[1, 2], [3, 4]]);
        assert_eq!(Vector::from([1, 2]), a[0].into());
        assert_eq!(Vector::from([3, 4]), a[1].into());
        assert_eq!(a, b);
    }

    #[test]
    fn add() {
        let a = Matrix::from([[1, 2], [3, 4]]);
        let b = Matrix::from([[5, 6], [7, 8]]);
        let c = Matrix::from([[6, 8], [10, 12]]);
        assert_eq!(a + b, c);
    }

    #[test]
    fn multiply() {
        let a = Matrix::from([[1, 4], [2, 5], [3, 6]]);
        let b = Matrix::from([[-1, 2, 3], [4, 5, 6]]);
        let c = Matrix::from([[12, 24], [32, 77]]);
        assert_eq!(a * b, c);
        let c = Matrix::from([[15, 22, 27], [18, 29, 36], [21, 36, 45]]);
        assert_eq!(b * a, c);
    }

    #[test]
    fn from_translation() {
        let a = Matrix::<_, 4, 4>::from_translation([1, 2, 3].into());
        let b = Matrix::from([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 2, 3, 1]]);
        assert_eq!(a, b);
    }

    #[test]
    fn from_scale() {
        let a = Matrix::<_, 4, 4>::from_scale(3);
        let b = Matrix::from([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]]);
        assert_eq!(a, b);
    }

    #[test]
    fn transpose_sqr() {
        let a = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
        let b = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert_eq!(a.transpose_sqr(), b);
    }
}
