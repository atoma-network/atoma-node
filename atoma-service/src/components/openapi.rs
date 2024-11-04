use axum::Router;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::handlers::chat_completions::{ChatCompletionsOpenApi, CHAT_COMPLETIONS_PATH};
use crate::handlers::embeddings::{EmbeddingsOpenApi, EMBEDDINGS_PATH};
use crate::handlers::image_generations::{ImageGenerationsOpenApi, IMAGE_GENERATIONS_PATH};
use crate::server::{AppState, HealthOpenApi, HEALTH_PATH};

pub fn openapi_routes() -> Router<AppState> {
    #[derive(OpenApi)]
    #[openapi(
        //modifiers(&SecurityAddon),
        nest(
            (path = HEALTH_PATH, api = HealthOpenApi),
            (path = CHAT_COMPLETIONS_PATH, api = ChatCompletionsOpenApi),
            (path = EMBEDDINGS_PATH, api = EmbeddingsOpenApi),
            (path = IMAGE_GENERATIONS_PATH, api = ImageGenerationsOpenApi),
        ),
        tags(
            (name = "health", description = "Health check"),
            (name = "chat", description = "Chat completions"),
            (name = "embeddings", description = "Embeddings"),
            (name = "images", description = "Image generations"),
        ),
        servers(
            (url = "http://localhost:8080"),
        )
    )]
    struct ApiDoc;

    // struct SecurityAddon;
    // impl Modify for SecurityAddon {
    //     fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
    //         if let Some(components) = openapi.components.as_mut() {
    //             components.add_security_scheme(
    //                 "bearerAuth",
    //                 SecurityScheme::Http(Http::new(HttpAuthScheme::Bearer)),
    //             )
    //         }
    //     }
    // }

    // Generate the OpenAPI spec and write it to a file
    #[cfg(debug_assertions)]
    {
        use std::fs;
        use std::path::Path;

        // Generate OpenAPI spec
        let spec =
            serde_yaml::to_string(&ApiDoc::openapi()).expect("Failed to serialize OpenAPI spec");

        // Ensure the docs directory exists
        let docs_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs");
        fs::create_dir_all(&docs_dir).expect("Failed to create docs directory");

        // Write the spec to the file
        let spec_path = docs_dir.join("openapi.yml");
        fs::write(&spec_path, spec).expect("Failed to write OpenAPI spec to file");

        println!("OpenAPI spec written to: {:?}", spec_path);
    }

    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
}
