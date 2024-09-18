use std::path::Path;

use config::Config;
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use url::{ParseError, Url};

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Configuration for Firebase.
pub struct SupabaseConfig {
    /// Url
    url: String,
    /// Anon key
    anon_key: String,
}

impl SupabaseConfig {
    /// Constructor
    pub fn new(url: String, anon_key: String) -> Self {
        Self { url, anon_key }
    }

    /// Get the https url from the config
    pub fn url(&self, protocol: &str) -> Result<Url, ParseError> {
        Url::parse(&format!("{protocol}://{}", self.url))
    }

    pub fn anon_key(&self) -> String {
        self.anon_key.clone()
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
            .get::<Self>("supabase")
            .expect("Failed to generated config file")
    }

    pub fn from_env_file() -> Self {
        dotenv().ok();

        let anon_key = std::env::var("SUPABASE_ANON_KEY")
            .unwrap_or_default()
            .parse()
            .unwrap();

        let url = std::env::var("SUPABASE_URL")
            .unwrap_or_default()
            .parse()
            .unwrap();

        Self { anon_key, url }
    }
}
