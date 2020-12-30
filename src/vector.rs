#[derive(Debug, Eq, PartialEq)]
pub struct MoveVector(pub i8, pub i8); // x, y

impl MoveVector {
    // Multiply a vector by a scalar magnitude
    pub fn times(&self, magnitude: u8) -> Self {
        let MoveVector(x, y) = self;

        MoveVector(x * magnitude as i8, y * magnitude as i8)
    }
}
