use std::sync::{RwLockReadGuard, RwLockWriteGuard};

pub trait DerefRead {
    type Error;
    type Inner;
    fn deref_read(&self) -> Result<RwLockReadGuard<Self::Inner>, Self::Error>;
}

pub trait DerefWrite {
    type Error;
    type Inner;
    fn deref_write(&self) -> Result<RwLockWriteGuard<Self::Inner>, Self::Error>;
}
