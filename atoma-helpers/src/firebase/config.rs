use atoma_types::SmallId;
use config::Config;
use dotenv::dotenv;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::path::Path;
use url::ParseError;

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Configuration for Firebase.
pub struct FirebaseConfig {
    /// Firebase api key
    api_key: String,
    /// Firebase url
    url: String,
    /// Small id
    small_id: SmallId,
}

impl FirebaseConfig {
    /// Constructor
    pub fn new(api_key: String, url: String, small_id: SmallId) -> Self {
        Self {
            api_key,
            url,
            small_id,
        }
    }

    /// Getter for `api_key`
    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    /// Get the firebase_url from the config
    pub fn url(&self) -> Result<Url, ParseError> {
        Url::parse(self.url.as_str())
    }

    pub fn small_id(&self) -> SmallId {
        self.small_id
    }

    /// Creates a new instance of `ModelsConfig` from a file path, containing the
    /// contents of a configuration file, with the above parameters specified.
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate firebase configuration file");
        config
            .get::<Self>("firebase")
            .expect("Failed to generated config file")
    }

    pub fn from_env_file() -> Self {
        dotenv().ok();

        let api_key = std::env::var("FIREBASE_API_KEY")
            .unwrap_or_default()
            .parse()
            .unwrap();

        let url = std::env::var("FIREBASE_URL")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let small_id = std::env::var("SMALL_ID")
            .unwrap_or_default()
            .parse()
            .unwrap();

        Self {
            api_key,
            url,
            small_id,
        }
    }
}
