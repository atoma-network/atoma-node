mod auth;
pub use auth::*;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Firebase {
    add_user_lock: Arc<Mutex<()>>,
}

impl Firebase {
    pub fn new() -> Self {
        Self {
            add_user_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn add_user(
        &self,
        email: String,
        password: String,
        api_key: String,
    ) -> Result<FirebaseAuth, FirebaseAuthError> {
        // This will prevent multiple calls to add_user from happening at the same time. Because in case the user doesn't exists it will trigger multiple signups.
        let _guard = self.add_user_lock.lock().await;
        let firebase_auth = FirebaseAuth::new(email, password, api_key).await?;
        Ok(firebase_auth)
    }
}
