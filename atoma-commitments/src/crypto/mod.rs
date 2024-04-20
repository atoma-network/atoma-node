mod keccak;

const COMMITMENT_SIZE: usize = 64;
const HASH_SIZE: usize = 32;

pub(crate) type Hash = [u8; HASH_SIZE];

pub struct Commitment([u8; COMMITMENT_SIZE]);

impl Commitment { 
    pub fn new(data: [u8; COMMITMENT_SIZE]) -> Self { 
        Self(data)
    }

    pub fn to_bytes(self) -> [u8; COMMITMENT_SIZE] { 
        self.0
    }
}

pub trait Hasher { 
    fn hash(data: &[u8]) -> [u8; HASH_SIZE];
}
