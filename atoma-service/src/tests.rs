mod middleware {
    use atoma_state::{
        types::{AtomaAtomaStateManagerEvent, Stack, Task},
        AtomaStateManager,
    };
    use atoma_sui::events::AtomaEvent;
    use atoma_utils::{hashing::blake2b_hash, test::POSTGRES_TEST_DB_URL};
    use axum::{
        body::Body, extract::Request, http::StatusCode, response::Response, routing::post, Router,
    };
    use base64::{prelude::BASE64_STANDARD, Engine};
    use flume::Sender;
    use serde_json::json;
    use serial_test::serial;
    use sqlx::PgPool;
    use std::{str::FromStr, sync::Arc};
    use sui_keys::keystore::{AccountKeystore, FileBasedKeystore};
    use sui_sdk::types::{
        base_types::SuiAddress,
        crypto::{EncodeDecodeBase64, PublicKey, Signature, SignatureScheme},
    };
    use tempfile::tempdir;
    use tokenizers::Tokenizer;
    use tokio::task::JoinHandle;
    use tower::Service;

    use crate::{
        handlers::{
            chat_completions::CHAT_COMPLETIONS_PATH, embeddings::EMBEDDINGS_PATH,
            image_generations::IMAGE_GENERATIONS_PATH,
        },
        middleware::{
            signature_verification_middleware, verify_stack_permissions, RequestMetadata,
            RequestType,
        },
        server::AppState,
    };

    const TEST_MESSAGE: &str = "Test message";

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

    async fn get_signature() -> String {
        let keystore = setup_keystore();
        let address = keystore.addresses()[0];
        let message = TEST_MESSAGE;
        let body_message = Body::from(message);

        // Create signature
        let body_message_bytes = axum::body::to_bytes(body_message, 1024)
            .await
            .expect("Failed to convert body to bytes");
        let blake2b_hash = blake2b_hash(body_message_bytes.as_ref());
        
        let signature = keystore
            .sign_hashed(&address, blake2b_hash.as_slice())
            .expect("Failed to sign message");
        BASE64_STANDARD.encode(signature.as_ref())
    }

    async fn load_tokenizer() -> Tokenizer {
        let url =
            "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/tokenizer.json";
        let tokenizer_json = reqwest::get(url).await.unwrap().text().await.unwrap();

        Tokenizer::from_str(&tokenizer_json).unwrap()
    }

    async fn truncate_tables() {
        let db = PgPool::connect(POSTGRES_TEST_DB_URL)
            .await
            .expect("Failed to connect to database");
        sqlx::query(
            "TRUNCATE TABLE 
                tasks,
                node_subscriptions,
                stacks,
                stack_settlement_tickets,
                stack_attestation_disputes
            CASCADE",
        )
        .execute(&db)
        .await
        .expect("Failed to truncate tables");
    }

    async fn setup_database(
        public_key: PublicKey,
    ) -> (
        JoinHandle<()>,
        Sender<AtomaAtomaStateManagerEvent>,
        tokio::sync::watch::Sender<bool>,
        Sender<AtomaEvent>,
    ) {
        let (_event_subscriber_sender, event_subscriber_receiver) = flume::unbounded();
        let (state_manager_sender, state_manager_receiver) = flume::unbounded();
        let state_manager = AtomaStateManager::new_from_url(
            POSTGRES_TEST_DB_URL,
            event_subscriber_receiver,
            state_manager_receiver,
        )
        .await
        .unwrap();
        let task = Task {
            task_small_id: 1,
            task_id: "1".to_string(),
            role: 0,
            model_name: Some("meta-llama/Llama-3.1-70B-Instruct".to_string()),
            security_level: 0,
            minimum_reputation_score: Some(100),
            is_deprecated: false,
            valid_until_epoch: Some(1),
            deprecated_at_epoch: Some(1),
        };
        state_manager.state.insert_new_task(task).await.unwrap();
        state_manager
            .state
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        let sui_address = SuiAddress::from(&public_key);
        let stack = Stack {
            owner_address: sui_address.to_string(),
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
        state_manager.state.insert_new_stack(stack).await.unwrap();
        let (shutdown_sender, shutdown_signal) = tokio::sync::watch::channel(false);
        let state_manager_handle = tokio::spawn(async move {
            state_manager.run(shutdown_signal).await.unwrap();
        });
        // NOTE: We don't need the event subscriber sender for the tests,
        // but we need to return it so the tests can send events to the state manager
        // otherwise the event subscriber will be dropped and the state manager shuts down
        (
            state_manager_handle,
            state_manager_sender,
            shutdown_sender,
            _event_subscriber_sender,
        )
    }

    async fn setup_app_state() -> (
        AppState,
        PublicKey,
        Signature,
        tokio::sync::watch::Sender<bool>,
        JoinHandle<()>,
        Sender<AtomaEvent>,
    ) {
        let keystore = setup_keystore();
        let models = vec![
            "meta-llama/Llama-3.1-70B-Instruct",
            "intfloat/multilingual-e5-large-instruct",
            "black-forest-labs/FLUX.1-schnell",
        ];
        let public_key = keystore.key_pairs().first().unwrap().public();
        let blake2b_hash = blake2b_hash(TEST_MESSAGE.as_bytes());
        let signature = keystore
            .sign_hashed(&keystore.addresses()[0], blake2b_hash.as_slice())
            .expect("Failed to sign message");
        let tokenizer = load_tokenizer().await;
        let (state_manager_handle, state_manager_sender, shutdown_sender, _event_subscriber_sender) =
            setup_database(public_key.clone()).await;
        let (stack_retrieve_sender, _) = tokio::sync::mpsc::unbounded_channel();
        (
            AppState {
                models: Arc::new(models.into_iter().map(|s| s.to_string()).collect()),
                tokenizers: Arc::new(vec![Arc::new(tokenizer.clone()), Arc::new(tokenizer)]),
                state_manager_sender,
                chat_completions_service_url: "".to_string(),
                embeddings_service_url: "".to_string(),
                image_generations_service_url: "".to_string(),
                keystore: Arc::new(keystore),
                address_index: 0,
                stack_retrieve_sender,
            },
            public_key,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        )
    }

    async fn test_handler(_: Request<Body>) -> Result<Response<Body>, StatusCode> {
        Ok(Response::new(Body::empty()))
    }

    #[test]
    fn test_request_metadata() {
        let request_metadata = RequestMetadata::default();

        assert_eq!(request_metadata.stack_small_id, 0);
        assert_eq!(request_metadata.estimated_total_compute_units, 0);
        assert_eq!(request_metadata.payload_hash, [0u8; 32]);

        let request_metadata = request_metadata.with_stack_info(1, 2);

        assert_eq!(request_metadata.stack_small_id, 1);
        assert_eq!(request_metadata.estimated_total_compute_units, 2);

        let request_metadata = request_metadata.with_payload_hash([3u8; 32]);

        assert_eq!(request_metadata.payload_hash, [3u8; 32]);
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

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
            "max_tokens": 100,
        });

        // Build request
        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", signature.encode_base64())
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

        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_missing_public_key() {
        let (app_state, _, _, shutdown_sender, state_manager_handle, _event_subscriber_sender) =
            setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            "max_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            // Intentionally omitting X-Signature header
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_unsupported_model() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

        let body = json!({
            "model": "unsupported-model",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            "max_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_invalid_messages_format() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": "not-an-array", // Invalid messages format
            "max_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri(CHAT_COMPLETIONS_PATH)
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new()
            .route(CHAT_COMPLETIONS_PATH, post(test_handler))
            .layer(axum::middleware::from_fn_with_state(
                app_state,
                verify_stack_permissions,
            ));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_missing_max_tokens() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            // Intentionally omitting max_tokens
        });

        let req = Request::builder()
            .method("POST")
            .uri(CHAT_COMPLETIONS_PATH)
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new()
            .route(CHAT_COMPLETIONS_PATH, post(test_handler))
            .layer(axum::middleware::from_fn_with_state(
                app_state,
                verify_stack_permissions,
            ));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_invalid_stack() {
        let (app_state, _, _, shutdown_sender, state_manager_handle, _event_subscriber_sender) =
            setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "system",
                "content": "You are a helpful assistant."
            }],
            "max_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", "invalid signature here") // Invalid signature
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new().route("/", post(test_handler)).layer(
            axum::middleware::from_fn_with_state(app_state, verify_stack_permissions),
        );

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_sets_request_metadata() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

        let body = json!({
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{
                "role": "system",
                "content": "You are a helpful assistant."
            }],
            "max_tokens": 100,
        });

        let req = Request::builder()
            .method("POST")
            .uri(CHAT_COMPLETIONS_PATH)
            .header("X-Signature", signature.encode_base64())
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
            assert!(metadata.estimated_total_compute_units > 0);

            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new()
            .route(CHAT_COMPLETIONS_PATH, post(check_metadata_handler))
            .layer(axum::middleware::from_fn_with_state(
                app_state,
                verify_stack_permissions,
            ));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);
        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_token_counting() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

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
            "max_tokens": 50,
        });

        let req = Request::builder()
            .method("POST")
            .uri(CHAT_COMPLETIONS_PATH)
            .header("X-Signature", signature.encode_base64())
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
            // 2. Should include max_tokens (50)
            // 3. Should include safety margins (3 tokens per message)
            assert!(metadata.estimated_total_compute_units > 50); // At least more than max_tokens

            // You could add more specific assertions based on your tokenizer's behavior
            // For example, if you know the exact token counts:
            // assert_eq!(metadata.estimated_total_tokens, expected_count);

            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new()
            .route(CHAT_COMPLETIONS_PATH, post(verify_token_count))
            .layer(axum::middleware::from_fn_with_state(
                app_state,
                verify_stack_permissions,
            ));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);
        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_success() {
        let signature = get_signature().await;

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", signature)
            .body(Body::from(TEST_MESSAGE))
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

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_missing_header() {
        let req = Request::builder()
            .method("POST")
            .uri("/")
            // Intentionally omitting X-Signature header
            .body(Body::from("Test message"))
            .unwrap();

        let mut app = Router::new()
            .route("/", post(test_handler))
            .layer(axum::middleware::from_fn(signature_verification_middleware));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_invalid_base64() {
        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", "not-valid-base64!")
            .body(Body::from("Test message"))
            .unwrap();

        let mut app = Router::new()
            .route("/", post(test_handler))
            .layer(axum::middleware::from_fn(signature_verification_middleware));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_invalid_signature() {
        let keystore = setup_keystore();
        let address = keystore.addresses()[0];
        let message = "Test message";

        // Sign a different message than what we'll send
        let different_message = "Different message";
        let blake2b_hash = blake2b_hash(different_message.as_bytes());
        let signature = keystore
            .sign_hashed(&address, blake2b_hash.as_slice())
            .expect("Failed to sign message");
        let signature_b64 = BASE64_STANDARD.encode(signature.as_ref());

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", signature_b64)
            .body(Body::from(message)) // Send original message with signature for different message
            .unwrap();

        let mut app = Router::new()
            .route("/", post(test_handler))
            .layer(axum::middleware::from_fn(signature_verification_middleware));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_empty_body() {
        let keystore = setup_keystore();
        let address = keystore.addresses()[0];

        // Sign empty message
        let blake2b_hash = blake2b_hash(&[]);

        let signature = keystore
            .sign_hashed(&address, blake2b_hash.as_slice())
            .expect("Failed to sign message");
        let signature_b64 = BASE64_STANDARD.encode(signature.as_ref());

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", signature_b64)
            .body(Body::empty())
            .unwrap();

        let mut app = Router::new()
            .route("/", post(test_handler))
            .layer(axum::middleware::from_fn(signature_verification_middleware));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_large_body() {
        let keystore = setup_keystore();
        let address = keystore.addresses()[0];

        // Create a body larger than MAX_BODY_SIZE
        let large_body = "x".repeat(2 * 1024 * 1024); // 2MB

        let blake2b_hash = blake2b_hash(large_body.as_bytes());

        let signature = keystore
            .sign_hashed(&address, blake2b_hash.as_slice())
            .expect("Failed to sign message");
        let signature_b64 = BASE64_STANDARD.encode(signature.as_ref());

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", signature_b64)
            .body(Body::from(large_body))
            .unwrap();

        let mut app = Router::new()
            .route("/", post(test_handler))
            .layer(axum::middleware::from_fn(signature_verification_middleware));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_preserves_body() {
        let keystore = setup_keystore();
        let address = keystore.addresses()[0];
        let message = "Test message";
        let body_message = Body::from(message);

        // Create signature
        let body_message_bytes = axum::body::to_bytes(body_message, 1024)
            .await
            .expect("Failed to convert body to bytes");
        let blake2b_hash = blake2b_hash(body_message_bytes.as_ref());
        
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

        // Custom handler that verifies the body content
        async fn verify_body_handler(req: Request<Body>) -> Result<Response<Body>, StatusCode> {
            let body_bytes = axum::body::to_bytes(req.into_body(), 1024)
                .await
                .expect("Failed to read body");
            assert_eq!(body_bytes, "Test message");
            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new()
            .route("/", post(verify_body_handler))
            .layer(axum::middleware::from_fn(signature_verification_middleware));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[serial]
    async fn test_signature_verification_updates_extensions() {
        let keystore = setup_keystore();
        let address = keystore.addresses()[0];
        let message = "Test message";
        let body_message = Body::from(message);

        // Create signature
        let body_message_bytes = axum::body::to_bytes(body_message, 1024)
            .await
            .expect("Failed to convert body to bytes");
        let blake2b_hash = blake2b_hash(body_message_bytes.as_ref());

        let signature = keystore
            .sign_hashed(&address, blake2b_hash.as_slice())
            .expect("Failed to sign message");
        let signature_b64 = BASE64_STANDARD.encode(signature.as_ref());

        // Create initial RequestMetadata with some existing values
        let initial_metadata = RequestMetadata {
            stack_small_id: 42,
            estimated_total_compute_units: 100,
            payload_hash: [0u8; 32],
            request_type: RequestType::ChatCompletions,
        };

        let mut req = Request::builder()
            .method("POST")
            .uri("/")
            .header("X-Signature", signature_b64)
            .body(Body::from(message))
            .unwrap();

        // Insert initial metadata
        req.extensions_mut().insert(initial_metadata);

        // Custom handler that verifies the extensions
        async fn verify_extensions_handler(
            req: Request<Body>,
        ) -> Result<Response<Body>, StatusCode> {
            let metadata = req
                .extensions()
                .get::<RequestMetadata>()
                .expect("Metadata should be set");

            // Verify that the payload hash was updated but other fields preserved
            assert_eq!(metadata.stack_small_id, 42);
            assert_eq!(metadata.estimated_total_compute_units, 100);
            assert_ne!(metadata.payload_hash, [0u8; 32]);
            assert_eq!(
                metadata.payload_hash,
                [
                    11, 151, 188, 173, 230, 19, 73, 18, 62, 134, 60, 28, 15, 134, 77, 75, 122, 182,
                    183, 33, 61, 174, 218, 225, 71, 33, 234, 229, 168, 253, 243, 109
                ]
            );

            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new()
            .route("/", post(verify_extensions_handler))
            .layer(axum::middleware::from_fn(signature_verification_middleware));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_embeddings() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

        // Test single string input
        let body = json!({
            "model": "intfloat/multilingual-e5-large-instruct",
            "input": "This is a test sentence for embedding.",
        });

        let req = Request::builder()
            .method("POST")
            .uri(EMBEDDINGS_PATH)
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        async fn verify_embeddings_compute_units(
            req: Request<Body>,
        ) -> Result<Response<Body>, StatusCode> {
            let metadata = req
                .extensions()
                .get::<RequestMetadata>()
                .expect("Metadata should be set");

            // Verify compute units are calculated correctly
            assert!(metadata.estimated_total_compute_units > 0);
            assert_eq!(metadata.request_type, RequestType::Embeddings);

            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new()
            .route(EMBEDDINGS_PATH, post(verify_embeddings_compute_units))
            .layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                verify_stack_permissions,
            ));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);

        // Test array of strings input
        let body = json!({
            "model": "intfloat/multilingual-e5-large-instruct",
            "input": ["First sentence.", "Second sentence.", "Third sentence."],
        });

        let req = Request::builder()
            .method("POST")
            .uri(EMBEDDINGS_PATH)
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);

        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_image_generation() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

        let body = json!({
            "model": "black-forest-labs/FLUX.1-schnell",
            "prompt": "A beautiful sunset over mountains",
            "size": "4x4",
            "n": 2
        });

        let req = Request::builder()
            .method("POST")
            .uri(IMAGE_GENERATIONS_PATH)
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        async fn verify_image_generation_compute_units(
            req: Request<Body>,
        ) -> Result<Response<Body>, StatusCode> {
            let metadata = req
                .extensions()
                .get::<RequestMetadata>()
                .expect("Metadata should be set");

            // For 4x4 image with n=2, should be 32 compute units (4 * 4 * 2)
            assert_eq!(metadata.estimated_total_compute_units, 32);
            assert_eq!(metadata.request_type, RequestType::ImageGenerations);

            Ok(Response::new(Body::empty()))
        }

        let mut app = Router::new()
            .route(
                IMAGE_GENERATIONS_PATH,
                post(verify_image_generation_compute_units),
            )
            .layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                verify_stack_permissions,
            ));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::OK);

        // Test invalid size format
        let body = json!({
            "model": "black-forest-labs/FLUX.1-schnell",
            "prompt": "A beautiful sunset over mountains",
            "size": "invalid",
            "n": 1
        });

        let req = Request::builder()
            .method("POST")
            .uri(IMAGE_GENERATIONS_PATH)
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        // Test missing n parameter
        let body = json!({
            "model": "black-forest-labs/FLUX.1-schnell",
            "prompt": "A beautiful sunset over mountains",
            "size": "4x4"
            // Missing "n" field
        });

        let req = Request::builder()
            .method("POST")
            .uri(IMAGE_GENERATIONS_PATH)
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_verify_stack_permissions_invalid_embeddings_input() {
        let (
            app_state,
            _,
            signature,
            shutdown_sender,
            state_manager_handle,
            _event_subscriber_sender,
        ) = setup_app_state().await;

        // Test with invalid input type (number instead of string or array)
        let body = json!({
            "model": "intfloat/multilingual-e5-large-instruct",
            "input": 123
        });

        let req = Request::builder()
            .method("POST")
            .uri(EMBEDDINGS_PATH)
            .header("X-Signature", signature.encode_base64())
            .header("X-Stack-Small-Id", "1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let mut app = Router::new()
            .route(EMBEDDINGS_PATH, post(test_handler))
            .layer(axum::middleware::from_fn_with_state(
                app_state,
                verify_stack_permissions,
            ));

        let response = app.call(req).await.expect("Failed to get response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        shutdown_sender.send(true).unwrap();
        state_manager_handle.await.unwrap();
        truncate_tables().await;
    }
}
