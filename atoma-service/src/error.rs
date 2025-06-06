use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Response structure for API errors
///
/// This struct is used to provide a consistent error response format across the API.
/// It wraps [`ErrorDetails`] in an `error` field to maintain a standard JSON structure
/// like `{"error": {...}}`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetails,
}

/// Details of an API error response
///
/// This struct contains the specific details of an error that occurred during
/// API request processing. It is wrapped in [`ErrorResponse`] to maintain a
/// consistent JSON structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetails {
    /// A machine-readable error code string (e.g., "MISSING_HEADER", "AUTH_ERROR")
    pub code: String,
    /// A human-readable error message describing what went wrong
    pub message: String,
}

/// Represents all possible errors that can occur within the Atoma service
///
/// This enum implements [`std::error::Error`] and provides structured error types
/// for various failure scenarios in the API. Each variant includes the number of
/// tokens that were processed before the error occurred.
#[derive(Debug, Error)]
pub enum AtomaServiceError {
    /// Error returned when a required HTTP header is missing from the request
    #[error("Missing required header: {header}")]
    MissingHeader {
        /// The name of the missing header
        header: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },

    /// Error returned when an HTTP header contains an invalid value
    #[error("Invalid header value: {message}")]
    InvalidHeader {
        /// Description of why the header value is invalid
        message: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },

    /// Error returned when the request body is malformed or contains invalid data
    #[error("Invalid request body: {message}")]
    InvalidBody {
        /// Description of why the request body is invalid
        message: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },

    /// Error returned when the underlying ML model encounters an error
    #[error("Model error: {model_error}")]
    ModelError {
        /// Description of the model error
        model_error: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },

    /// Error returned when authentication fails
    #[error("Authentication error: {auth_error}")]
    AuthError {
        /// Description of the authentication error
        auth_error: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },

    /// Error returned for unexpected internal server errors
    #[error("Internal server error: {message}")]
    InternalError {
        /// Description of the internal error
        message: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },

    /// Error returned when the stack is locked
    #[error("Stack is locked: {message}")]
    LockedStackError {
        /// Description of why the stack is locked
        message: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },

    /// Error returned when the stack is unavailable
    #[error("Stack is unavailable: {message}")]
    UnavailableStackError {
        /// Description of why the stack is unavailable
        message: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },

    /// Error returned when the chat completions service is unavailable
    #[error("Chat completions service is unavailable: {message}")]
    ChatCompletionsServiceUnavailable {
        /// Description of why the chat completions service is unavailable
        message: String,
        /// The endpoint that the error occurred on
        endpoint: String,
    },
}

impl AtomaServiceError {
    /// Returns the machine-readable error code for this error type
    ///
    /// Each variant of [`AtomaServiceError`] maps to a specific error code string that can be used
    /// by API clients to programmatically handle different error cases. These codes are consistent
    /// across API responses and are included in the [`ErrorDetails`] structure.
    ///
    /// # Returns
    ///
    /// A static string representing the error code, such as:
    /// - `"MISSING_HEADER"` for missing required headers
    /// - `"INVALID_HEADER"` for invalid header values
    /// - `"INVALID_BODY"` for malformed request bodies
    /// - `"MODEL_ERROR"` for ML model errors
    /// - `"AUTH_ERROR"` for authentication failures
    /// - `"INTERNAL_ERROR"` for unexpected server errors
    const fn error_code(&self) -> &'static str {
        match self {
            Self::MissingHeader { .. } => "MISSING_HEADER",
            Self::InvalidHeader { .. } => "INVALID_HEADER",
            Self::InvalidBody { .. } => "INVALID_BODY",
            Self::ModelError { .. } => "MODEL_ERROR",
            Self::AuthError { .. } => "AUTH_ERROR",
            Self::InternalError { .. } => "INTERNAL_ERROR",
            Self::LockedStackError { .. } => "LOCKED_STACK_ERROR",
            Self::UnavailableStackError { .. } => "UNAVAILABLE_STACK_ERROR",
            Self::ChatCompletionsServiceUnavailable { .. } => {
                "CHAT_COMPLETIONS_SERVICE_UNAVAILABLE"
            }
        }
    }

    /// Returns a user-friendly error message for API responses
    ///
    /// This method generates human-readable error messages that are suitable for
    /// displaying to API clients. The messages are more concise than the full
    /// error details and exclude sensitive internal information.
    ///
    /// # Returns
    ///
    /// A `String` containing a user-friendly error message based on the error type:
    /// - For missing headers: Specifies which header is missing
    /// - For invalid headers: A generic invalid header message
    /// - For invalid body: Includes the specific validation error
    /// - For model errors: Includes the model-specific error message
    /// - For auth errors: A generic authentication failure message
    /// - For internal errors: A generic server error message
    fn client_message(&self) -> String {
        match self {
            Self::MissingHeader { header, .. } => format!("Missing required header: {}", header),
            Self::InvalidHeader { .. } => "Invalid header value provided".to_string(),
            Self::InvalidBody { message, .. } => format!("Invalid request body: {}", message),
            Self::ModelError { model_error, .. } => format!("Model error: {}", model_error),
            Self::AuthError { .. } => "Authentication failed".to_string(),
            Self::InternalError { .. } => "Internal server error occurred".to_string(),
            Self::LockedStackError { .. } => "Stack is locked".to_string(),
            Self::UnavailableStackError { .. } => "Stack is unavailable".to_string(),
            Self::ChatCompletionsServiceUnavailable { .. } => {
                "Chat completions service is unavailable".to_string()
            }
        }
    }

    /// Returns the HTTP status code associated with this error
    ///
    /// Maps each error variant to an appropriate HTTP status code:
    /// - `400 Bad Request` for invalid inputs (missing/invalid headers, invalid body, model errors)
    /// - `401 Unauthorized` for authentication failures
    /// - `500 Internal Server Error` for unexpected server errors
    ///
    /// # Returns
    ///
    /// An [`axum::http::StatusCode`] representing the appropriate HTTP response code for this error
    #[must_use]
    pub const fn status_code(&self) -> StatusCode {
        match self {
            Self::MissingHeader { .. }
            | Self::InvalidHeader { .. }
            | Self::InvalidBody { .. }
            | Self::ModelError { .. } => StatusCode::BAD_REQUEST,
            Self::AuthError { .. } => StatusCode::UNAUTHORIZED,
            Self::InternalError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::LockedStackError { .. } => StatusCode::LOCKED,
            Self::UnavailableStackError { .. } => StatusCode::TOO_EARLY,
            Self::ChatCompletionsServiceUnavailable { .. } => StatusCode::TOO_MANY_REQUESTS,
        }
    }

    /// Returns the endpoint where the error occurred
    ///
    /// This method provides access to the endpoint path across all error variants,
    /// which is useful for error tracking, logging, and debugging purposes.
    ///
    /// # Returns
    ///
    /// A `String` containing the API endpoint path where the error was encountered.
    #[must_use]
    pub fn get_endpoint(&self, _endpoint: &str) -> String {
        match self {
            Self::MissingHeader { endpoint, .. }
            | Self::InvalidHeader { endpoint, .. }
            | Self::InvalidBody { endpoint, .. }
            | Self::ModelError { endpoint, .. }
            | Self::AuthError { endpoint, .. }
            | Self::InternalError { endpoint, .. }
            | Self::LockedStackError { endpoint, .. }
            | Self::UnavailableStackError { endpoint, .. }
            | Self::ChatCompletionsServiceUnavailable { endpoint, .. } => endpoint.clone(),
        }
    }

    /// Returns the full error message with details
    ///
    /// Unlike [`client_message`], this method returns the complete error message including
    /// internal details. This is suitable for internal logging and debugging, but should
    /// not be exposed directly to API clients.
    ///
    /// # Returns
    ///
    /// A `String` containing the detailed error message that includes:
    /// - For missing headers: The name of the missing header
    /// - For invalid headers: The specific validation error
    /// - For invalid body: The detailed validation message
    /// - For model errors: The complete model error message
    /// - For auth errors: The specific authentication failure reason
    /// - For internal errors: The detailed internal error message
    fn message(&self) -> String {
        match self {
            Self::MissingHeader { header, .. } => format!("Missing required header: {}", header),
            Self::InvalidHeader { message, .. } => format!("Invalid header value: {}", message),
            Self::InvalidBody { message, .. } => format!("Invalid request body: {}", message),
            Self::ModelError { model_error, .. } => format!("Model error: {}", model_error),
            Self::AuthError { auth_error, .. } => format!("Authentication error: {}", auth_error),
            Self::InternalError { message, .. } => format!("Internal server error: {}", message),
            Self::LockedStackError { message, .. } => format!("Stack is locked: {}", message),
            Self::UnavailableStackError { message, .. } => {
                format!("Stack is unavailable: {}", message)
            }
            Self::ChatCompletionsServiceUnavailable { message, .. } => {
                format!("Chat completions service is unavailable: {}", message)
            }
        }
    }
}

impl IntoResponse for AtomaServiceError {
    fn into_response(self) -> Response {
        tracing::error!(
            target = "atoma-service",
            event = "error_occurred",
            endpoint = self.get_endpoint(""),
            error = %self.message(),
        );
        let error_response = ErrorResponse {
            error: ErrorDetails {
                code: self.error_code().to_string(),
                message: self.client_message(),
            },
        };
        (self.status_code(), Json(error_response)).into_response()
    }
}
