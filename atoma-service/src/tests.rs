mod middleware {
    use atoma_state::{
        types::{Stack, Task},
        SqlitePool, StateManager,
    };
    use axum::{
        body::Body, extract::Request, http::StatusCode, response::Response, routing::post, Router,
    };
    use base64::{prelude::BASE64_STANDARD, Engine};
    use blake2::{digest::generic_array::GenericArray, Digest};
    use hex::ToHex;
    use p256::U32;
    use reqwest::Client;
    use serde_json::json;
    use serial_test::serial;
    use std::{path::PathBuf, str::FromStr, sync::Arc};
    use sui_keys::keystore::{AccountKeystore, FileBasedKeystore};
    use sui_sdk::types::crypto::{EncodeDecodeBase64, PublicKey, SignatureScheme};
    use tempfile::tempdir;
    use tokenizers::Tokenizer;
    use tower::Service;

    use crate::{
        middleware::{
            signature_verification_middleware, verify_stack_permissions, RequestMetadata,
        },
        server::AppState,
    };

    fn setup_keystore() -> FileBasedKeystore {
        let temp_dir = tempdir().unwrap();
        let keystore_path = temp_dir.path().join("keystore");
        let mut keystore = FileBasedKeystore::new(&keystore_path).unwrap();
        keystore
            .generate_and_add_new_key(
                SignatureScheme::ED25519,
                Some("test".to_string()),
                None,
                None,
            )
            .unwrap();
        keystore
    }

    async fn load_tokenizer() -> Tokenizer {
        let url =
            "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/tokenizer.json";
        let tokenizer_json = reqwest::get(url).await.unwrap().text().await.unwrap();

        Tokenizer::from_str(&tokenizer_json).unwrap()
    }

    async fn setup_database(public_key: String) -> (SqlitePool, PathBuf) {
        let db_path = std::path::Path::new("./db_path").to_path_buf();

        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&db_path)
            .unwrap();

        let database_url = format!("sqlite:{}", db_path.to_str().unwrap());
        let state_manager = StateManager::new_from_url(database_url).await.unwrap();
        let task = Task {
            task_small_id: 1,
            task_id: "1".to_string(),
            role: 0,
            model_name: Some("meta-llama/Llama-3.1-70B-Instruct".to_string()),
            optimizations: "".to_string(),
            security_level: 0,
            task_metrics_compute_unit: 1,
            task_metrics_time_unit: Some(1),
            task_metrics_value: Some(10000),
            minimum_reputation_score: Some(100),
            is_deprecated: false,
            valid_until_epoch: Some(1),
            deprecated_at_epoch: Some(1),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        let stack = Stack {
            owner_address: format!("0x00{}", public_key),
            stack_small_id: 1,
            stack_id: "1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 600,
            price: 1,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 1,
        };
        state_manager.insert_new_stack(stack).await.unwrap();
        (state_manager.into_pool(), db_path)
    }

    async fn setup_app_state() -> (AppState, PublicKey, PathBuf) {
        let keystore = setup_keystore();
        let models = vec!["meta-llama/Llama-3.1-70B-Instruct"];
        let public_key = keystore.key_pairs().first().unwrap().public();
        let tokenizer = load_tokenizer().await;
        let (state, db_path) = setup_database(public_key.encode_hex()).await;
        (
            AppState {
                models: Arc::new(models.into_iter().map(|s| s.to_string()).collect()),
                tokenizer: Arc::new(tokenizer),
                state,
                inference_service_client: Client::new(),
                keystore: Arc::new(keystore),
                address_index: 0,
            },
            public_key,
            db_path,
        )
    }

    async fn test_handler(_: Request<Body>) -> Result<Response<Body>, StatusCode> {
        Ok(Response::new(Body::empty()))
    }

    #[test]
    fn test_request_metadata() {
        let request_metadata = RequestMetadata::default();

        assert_eq!(request_metadata.stack_small_id, 0);
        assert_eq!(request_metadata.estimated_total_tokens, 0);
        assert_eq!(request_metadata.payload_hash, [0u8; 32]);

        let request_metadata = request_metadata.with_stack_info(1, 2);

        assert_eq!(request_metadata.stack_small_id, 1);
        assert_eq!(request_metadata.estimated_total_tokens, 2);

        let request_metadata = request_metadata.with_payload_hash([3u8; 32]);

        assert_eq!(request_metadata.payload_hash, [3u8; 32]);
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions() {
        let (app_state, public_key, db_path) = setup_app_state().await;

        // Create request body
        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of the moon?"
            }],
            "max_completion_tokens": 100,
        });

        // Build request
        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-PublicKey", public_key.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        // Build a router with your middleware applied
        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state.clone(), verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");

        assert_eq!(response.status(), StatusCode::OK);
        std::fs::remove_file(db_path).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_missing_public_key() {
        let (app_state, _, db_path) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            "max_completion_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            // Intentionally omitting X-PublicKey header
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        std::fs::remove_file(db_path).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_unsupported_model() {
        let (app_state, public_key, db_path) = setup_app_state().await;

        let body = json!({
            "model": "unsupported-model",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            "max_completion_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-PublicKey", public_key.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        std::fs::remove_file(db_path).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_invalid_messages_format() {
        let (app_state, public_key, db_path) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": "not-an-array", // Invalid messages format
            "max_completion_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-PublicKey", public_key.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        std::fs::remove_file(db_path).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_missing_max_completion_tokens() {
        let (app_state, public_key, db_path) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            // Intentionally omitting max_completion_tokens
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-PublicKey", public_key.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        std::fs::remove_file(db_path).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_invalid_stack() {
        let (app_state, _, db_path) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "system",
                "content": "You are a helpful assistant."
            }],
            "max_completion_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-PublicKey", "invalid_key_here") // Invalid public key
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        std::fs::remove_file(db_path).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_sets_request_metadata() {
        let (app_state, public_key, db_path) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "system",
                "content": "You are a helpful assistant."
            }],
            "max_completion_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-PublicKey", public_key.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        // Create a custom handler that checks the extensions
        async fn check_metadata_handler(req: Request<Body>) -> Result<Response<Body>, StatusCode> {
            let metadata = req
                .extensions()
                .get::<RequestMetadata>()
                .expect("Metadata should be set");

            assert_eq!(metadata.stack_small_id, 1);
            // The exact token count will depend on your tokenizer, but we can verify it's non-zero
            assert!(metadata.estimated_total_tokens > 0);

            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new()
            .route("/", post(check_metadata_handler))
            .layer(axum::middleware::from_fn_with_state(
                app_state,
                verify_stack_permissions,
            ));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);
        std::fs::remove_file(db_path).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_token_counting() {
        let (app_state, public_key, db_path) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "Short message"
                },
                {
                    "role": "user",
                    "content": "Longer message with more tokens to count"
                }
            ],
            "max_completion_tokens": 50,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-PublicKey", public_key.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        async fn verify_token_count(req: Request<Body>) -> Result<Response<Body>, StatusCode> {
            let metadata = req
                .extensions()
                .get::<RequestMetadata>()
                .expect("Metadata should be set");

            // Verify token counting logic:
            // 1. Should include tokens from both messages
            // 2. Should include max_completion_tokens (50)
            // 3. Should include safety margins (3 tokens per message)
            assert!(metadata.estimated_total_tokens > 50); // At least more than max_completion_tokens

            // You could add more specific assertions based on your tokenizer's behavior
            // For example, if you know the exact token counts:
            // assert_eq!(metadata.estimated_total_tokens, expected_count);

            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new().route("/", post(verify_token_count)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);
        std::fs::remove_file(db_path).unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_success() {
        let keystore = setup_keystore();
        let address = keystore.addresses()[0];
        let message = "Test message";
        let body_message = Body::from(message);

        // Create signature
        let mut blake2b = blake2::Blake2b::new();
        let body_message_bytes = axum::body::to_bytes(body_message, 1024)
            .await
            .expect("Failed to convert body to bytes");
        blake2b.update(body_message_bytes);
        let blake2b_hash: GenericArray<u8, U32> = blake2b.finalize();

        let signature = keystore
            .sign_hashed(&address, blake2b_hash.as_slice())
            .expect("Failed to sign message");
        let signature_b64 = BASE64_STANDARD.encode(signature.as_ref());

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", signature_b64)
            .body(Body::from(message))
            .unwrap();

        async fn check_metadata_handler(req: Request<Body>) -> Result<Response<Body>, StatusCode> {
            let metadata = req
                .extensions()
                .get::<RequestMetadata>()
                .expect("Metadata should be set");
            assert_ne!(metadata.payload_hash, [0u8; 32]); // Hash should be set
            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new()
            .route("/", post(check_metadata_handler))
            .layer(axum::middleware::from_fn(signature_verification_middleware));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);
    }
}
