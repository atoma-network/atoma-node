use serde::Deserialize;

pub mod subscriber;

#[derive(Debug, Deserialize)]
pub enum RequestType { 
    TextToImage,
    TextToText,
}