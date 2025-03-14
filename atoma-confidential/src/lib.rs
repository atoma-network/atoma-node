#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]

pub mod key_management;
pub mod nvml_cc;
pub mod service;
pub mod types;

pub use service::AtomaConfidentialCompute;
