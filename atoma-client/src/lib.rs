mod client;
mod config;

pub use client::{AtomaSuiClient, AtomaSuiClientError};

pub struct SuiConstants { 
    package_id: String,
    module_id: String,
    method: String,
}