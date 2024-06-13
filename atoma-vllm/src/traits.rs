use std::sync::{RwLockReadGuard, RwLockWriteGuard};

pub trait BlockReadLock {
    type Error;
    type Inner;
    fn read_lock(&self) -> Result<RwLockReadGuard<Self::Inner>, Self::Error>;
}

pub trait BlockWriteLock {
    type Error;
    type Inner;
    fn write_lock(&self) -> Result<RwLockWriteGuard<Self::Inner>, Self::Error>;
}
