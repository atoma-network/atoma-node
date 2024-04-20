const COMMITMENT_SIZE: usize = 64;

pub struct Commitment([u8; COMMITMENT_SIZE]);

impl Commitment {
    pub fn new(data: [u8; COMMITMENT_SIZE]) -> Self {
        Self(data)
    }

    pub fn to_bytes(self) -> [u8; COMMITMENT_SIZE] {
        self.0
    }
}

