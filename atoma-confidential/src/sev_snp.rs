use crate::ToBytes;
use dcap_rs::types::quotes::{body::QuoteBody, version_4::QuoteV4};
use sev::{
    attestation::{
        AttestationReport, 
        AttestationReportSignature
    },
    error::SevError,
    firmware::Firmware,
};

use thiserror::Error;


pub const SEV_SNP_REPORT_DATA_SIZE: usize = 64;

type Result<T> = std::result::Result<T, SevError>;


