use serde::{Deserialize, Serialize};
use std::str::FromStr;
use sui_sdk::types::base_types::SuiAddress;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, SuiEventParseError>;

/// Represents the various events that can be emitted by the Atoma contract on the Sui blockchain.
///
/// This enum encapsulates all possible events across different modules of the Atoma system,
/// including database operations, settlement processes, and specific AI task events.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum AtomaEventIdentifier {
    /// Events related to the database (Db) module:
    ///
    /// Emitted when the Atoma contract is first published.
    PublishedEvent,
    /// Emitted when a new node is registered in the network.
    NodeRegisteredEvent,
    /// Emitted when a node subscribes to a specific AI model.
    NodeSubscribedToModelEvent,
    /// Emitted when a node subscribes to a specific task.
    NodeSubscribedToTaskEvent,
    /// Emitted when a node updates its subscription to a task.
    NodeSubscriptionUpdatedEvent,
    /// Emitted when a node unsubscribes from a task.
    NodeUnsubscribedFromTaskEvent,
    /// Emitted when a new task is registered in the network.
    TaskRegisteredEvent,
    /// Emitted when a task is marked as deprecated.
    TaskDeprecationEvent,
    /// Emitted when a task is permanently removed from the network.
    TaskRemovedEvent,
    /// Emitted when a new stack (collection of tasks) is created.
    StackCreatedEvent,
    /// Emitted when there's an attempt to settle a stack.
    StackTrySettleEvent,
    /// Emitted when a new attestation is made for a stack settlement.
    NewStackSettlementAttestationEvent,
    /// Emitted when a settlement ticket is issued for a stack.
    StackSettlementTicketEvent,
    /// Emitted when a stack settlement ticket is claimed.
    StackSettlementTicketClaimedEvent,
    /// Emitted when there's a dispute in the attestation process for a stack.
    StackAttestationDisputeEvent,

    /// Events related to the settlement module:
    ///
    /// Emitted when the first submission is made in a settlement process.
    FirstSubmissionEvent,
    /// Emitted when a dispute occurs during settlement.
    DisputeEvent,
    /// Emitted when new nodes are sampled for a process (e.g., for attestation).
    NewlySampledNodesEvent,
    /// Emitted when a settlement process is successfully completed.
    SettledEvent,
    /// Emitted when a settlement process needs to be retried.
    RetrySettlementEvent,

    /// Events related to the gate module (specific AI tasks):
    ///
    /// Emitted when a text-to-image prompt is processed.
    Text2ImagePromptEvent,
    /// Emitted when a text-to-text prompt is processed.
    Text2TextPromptEvent,

    /// Events related to node public key rotation with TEE:
    ///
    /// Emitted when a node's public key is rotated.
    NewKeyRotationEvent,
    /// Emitted when a node's key rotation remote attestation is verified.
    NodePublicKeyCommittmentEvent,
}

impl FromStr for AtomaEventIdentifier {
    type Err = SuiEventParseError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "PublishedEvent" => Ok(Self::PublishedEvent),
            "NodeRegisteredEvent" => Ok(Self::NodeRegisteredEvent),
            "NodeSubscribedToModelEvent" => Ok(Self::NodeSubscribedToModelEvent),
            "NodeSubscribedToTaskEvent" => Ok(Self::NodeSubscribedToTaskEvent),
            "NodeUnsubscribedFromTaskEvent" => Ok(Self::NodeUnsubscribedFromTaskEvent),
            "TaskRegisteredEvent" => Ok(Self::TaskRegisteredEvent),
            "TaskDeprecationEvent" => Ok(Self::TaskDeprecationEvent),
            "TaskRemovedEvent" => Ok(Self::TaskRemovedEvent),
            "StackCreatedEvent" => Ok(Self::StackCreatedEvent),
            "StackTrySettleEvent" => Ok(Self::StackTrySettleEvent),
            "NewStackSettlementAttestationEvent" => Ok(Self::NewStackSettlementAttestationEvent),
            "StackSettlementTicketEvent" => Ok(Self::StackSettlementTicketEvent),
            "StackSettlementTicketClaimedEvent" => Ok(Self::StackSettlementTicketClaimedEvent),
            "StackAttestationDisputeEvent" => Ok(Self::StackAttestationDisputeEvent),
            "FirstSubmissionEvent" => Ok(Self::FirstSubmissionEvent),
            "DisputeEvent" => Ok(Self::DisputeEvent),
            "NewlySampledNodesEvent" => Ok(Self::NewlySampledNodesEvent),
            "SettledEvent" => Ok(Self::SettledEvent),
            "RetrySettlementEvent" => Ok(Self::RetrySettlementEvent),
            "Text2ImagePromptEvent" => Ok(Self::Text2ImagePromptEvent),
            "Text2TextPromptEvent" => Ok(Self::Text2TextPromptEvent),
            "NodeSubscriptionUpdatedEvent" => Ok(Self::NodeSubscriptionUpdatedEvent),
            "NewKeyRotationEvent" => Ok(Self::NewKeyRotationEvent),
            "NodePublicKeyCommittmentEvent" => Ok(Self::NodePublicKeyCommittmentEvent),
            _ => Err(SuiEventParseError::UnknownEvent(s.to_string())),
        }
    }
}

/// Represents the various events that can occur within the Atoma network.
///
/// This enum encapsulates all possible events emitted by the Atoma contract,
/// allowing for structured handling of different event types in the system.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum AtomaEvent {
    /// An event emitted when the Atoma contract is first published.
    PublishedEvent(PublishedEvent),

    /// An event emitted when a new node is registered in the Atoma network.
    NodeRegisteredEvent((NodeRegisteredEvent, SuiAddress)),

    /// An event emitted when a node subscribes to a specific AI model.
    NodeSubscribedToModelEvent(NodeSubscribedToModelEvent),

    /// An event emitted when a node subscribes to a specific task.
    NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent),

    /// An event emitted when a node updates its subscription to a task.
    NodeSubscriptionUpdatedEvent(NodeSubscriptionUpdatedEvent),

    /// An event emitted when a node unsubscribes from a specific task.
    NodeUnsubscribedFromTaskEvent(NodeUnsubscribedFromTaskEvent),

    /// An event emitted when a new task is registered in the Atoma network.
    TaskRegisteredEvent(TaskRegisteredEvent),

    /// An event emitted when a task is marked as deprecated.
    TaskDeprecationEvent(TaskDeprecationEvent),

    /// An event emitted when a task is permanently removed from the Atoma network.
    TaskRemovedEvent(TaskRemovedEvent),

    /// An event emitted when a new stack (collection of tasks) is created.
    StackCreatedEvent((StackCreatedEvent, Option<u64>)),

    /// An event emitted when a stack is created and updated simultaneously with already computed compute units.
    StackCreateAndUpdateEvent(StackCreateAndUpdateEvent),

    /// An event emitted when there's an attempt to settle a stack.
    StackTrySettleEvent((StackTrySettleEvent, Option<u64>)),

    /// An event emitted when a new attestation is made for a stack settlement.
    NewStackSettlementAttestationEvent(NewStackSettlementAttestationEvent),

    /// An event emitted when a settlement ticket is issued for a stack.
    StackSettlementTicketEvent(StackSettlementTicketEvent),

    /// An event emitted when a stack settlement ticket is claimed.
    StackSettlementTicketClaimedEvent(StackSettlementTicketClaimedEvent),

    /// An event emitted when there's a dispute in the attestation process for a stack.
    StackAttestationDisputeEvent(StackAttestationDisputeEvent),

    /// An event emitted when the first submission is made in a settlement process.
    FirstSubmissionEvent(FirstSubmissionEvent),

    /// An event emitted when a dispute occurs during settlement.
    DisputeEvent(DisputeEvent),

    /// An event emitted when new nodes are sampled for a process (e.g., for attestation).
    NewlySampledNodesEvent(NewlySampledNodesEvent),

    /// An event emitted when a settlement process is successfully completed.
    SettledEvent(SettledEvent),

    /// An event emitted when a settlement process needs to be retried.
    RetrySettlementEvent(RetrySettlementEvent),

    /// An event emitted when a text-to-image prompt is processed.
    Text2ImagePromptEvent(Text2ImagePromptEvent),

    /// An event emitted when a text-to-text prompt is processed.
    Text2TextPromptEvent(Text2TextPromptEvent),

    /// An event emitted when Atoma's smart contract requests new node key rotation.
    NewKeyRotationEvent(NewKeyRotationEvent),

    /// An event emitted when a node's key rotation remote attestation is verified successfully.
    NodePublicKeyCommittmentEvent(NodePublicKeyCommittmentEvent),
}

fn deserialize_string_to_u64<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::de::Deserializer<'de>,
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<T>().map_err(serde::de::Error::custom)
}

/// Represents an event that is emitted when the Atoma contract is first published.
///
/// This event contains information about the newly published AtomaDb object id and
/// the associated manager badge id.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PublishedEvent {
    /// The object id of the AtomaDb.
    pub db: String,

    /// The identifier of the manager badge associated with the Atoma contract.
    /// This badge grants certain administrative privileges for the owner.
    pub manager_badge: String,
}

/// Represents an event that is emitted when a new node is registered in the Atoma network.
///
/// This event contains information about the newly registered node, including its unique
/// badge identifier and small ID within the network.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeRegisteredEvent {
    /// The unique identifier of the badge associated with the registered node.
    /// This badge serves as a proof of registration and grants node privileges within the network.
    pub badge_id: String,

    /// The small ID assigned to the node within the Atoma network.
    /// This ID is used for efficient referencing of the node in various operations and events.
    pub node_small_id: NodeSmallId,
}

/// Represents an event that is emitted when a node subscribes to a specific model in the Atoma network.
///
/// This event contains information about the subscribing node, the model it's subscribing to,
/// and the echelon (performance tier) of the subscription.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeSubscribedToModelEvent {
    /// The small ID of the node that is subscribing to the model.
    /// This is a compact identifier for the node within the Atoma network.
    pub node_small_id: NodeSmallId,

    /// The name of the model that the node is subscribing to.
    /// This field represents the name of the AI models avaiable in the network
    /// (which is compatible with HuggingFace's model naming convention).
    pub model_name: String,

    /// The echelon ID representing the performance tier or capability level
    /// at which the node is subscribing to the model.
    /// Different echelons may have different computational requirements or privileges.
    pub echelon_id: EchelonId,
}

/// Represents an event that is emitted when a node subscribes to a specific task in the Atoma network.
///
/// This event contains information about the subscribing node, the task it's subscribing to,
/// and the price per compute unit that the node is offering for this task.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeSubscribedToTaskEvent {
    /// The small ID of the task that the node is subscribing to.
    /// This is a compact identifier for the task within the Atoma network.
    pub task_small_id: TaskSmallId,

    /// The small ID of the node that is subscribing to the task.
    /// This is a compact identifier for the node within the Atoma network.
    pub node_small_id: NodeSmallId,

    /// The price per compute unit that the node is offering for this task.
    /// This represents the cost in Atoma's native currency for each unit of computation
    /// that the node will perform for this task.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub price_per_one_million_compute_units: u64,

    /// The maximum number of compute units that the node is willing to process for this task.
    /// This limits the amount of resources the node will commit to processing the task.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub max_num_compute_units: u64,
}

/// Represents an event that is emitted when a node updates its subscription to a task in the Atoma network.
///
/// This event contains information about the node, the task it's updating its subscription to,
/// and the new price per compute unit that the node is offering for this task.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeSubscriptionUpdatedEvent {
    /// The small ID of the task that the node is updating its subscription to.
    /// This is a compact identifier for the task within the Atoma network.
    pub task_small_id: TaskSmallId,

    /// The small ID of the node that is updating its subscription to the task.
    /// This is a compact identifier for the node within the Atoma network.
    pub node_small_id: NodeSmallId,

    /// The new price per compute unit that the node is offering for this task.
    /// This represents the cost in Atoma's native currency for each unit of computation
    /// that the node will perform for this task.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub price_per_one_million_compute_units: u64,

    /// The maximum number of compute units that the node is willing to process for this task.
    /// This limits the amount of resources the node will commit to processing the task.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub max_num_compute_units: u64,
}

/// Represents an event that is emitted when a node unsubscribes from a specific task in the Atoma network.
///
/// This event contains information about the unsubscribing node and the task it's unsubscribing from.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeUnsubscribedFromTaskEvent {
    /// The small ID of the task that the node is unsubscribing from.
    /// This is a compact identifier for the task within the Atoma network.
    pub task_small_id: TaskSmallId,

    /// The small ID of the node that is unsubscribing from the task.
    /// This is a compact identifier for the node within the Atoma network.
    pub node_small_id: NodeSmallId,
}

/// Represents an event that is emitted when a new task is registered in the Atoma network.
///
/// This event contains comprehensive information about the newly registered task, including its
/// identifiers, role, associated model, security level and minimum reputation requirements.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskRegisteredEvent {
    /// The unique identifier of the task.
    /// This is typically a longer, more descriptive ID for the task.
    pub task_id: String,

    /// The small ID assigned to the task within the Atoma network.
    /// This ID is used for efficient referencing of the task in various operations and events.
    pub task_small_id: TaskSmallId,

    /// The role of the task in the Atoma network.
    /// This defines the task's function or purpose within the system.
    pub role: TaskRole,

    /// The name of the AI model associated with this task, if applicable.
    /// This field is optional as not all tasks may be tied to a specific model.
    pub model_name: Option<String>,

    /// The security level required for this task.
    /// Higher values typically indicate stricter security measures or clearance levels.
    pub security_level: SecurityLevel,

    /// The minimum reputation score required for a node to work on this task, if applicable.
    /// This helps ensure that only sufficiently trusted nodes can participate in certain tasks.
    pub minimum_reputation_score: Option<u8>,
}

/// Represents an event that is emitted when a task is deprecated in the Atoma network.
///
/// This event contains information about the deprecated task, including its identifiers
/// and the epoch at which it was deprecated.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskDeprecationEvent {
    /// The unique identifier of the deprecated task.
    /// This is typically a longer, more descriptive ID for the task.
    pub task_id: String,

    /// The small ID of the deprecated task within the Atoma network.
    /// This ID is used for efficient referencing of the task in various operations and events.
    pub task_small_id: TaskSmallId,

    /// The epoch at which the task was deprecated.
    /// An epoch represents a specific point in time or a block height in the network.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub epoch: u64,
}

/// Represents an event that is emitted when a task is permanently removed from the Atoma network.
///
/// This event contains information about the removed task, including its identifiers
/// and the epoch at which it was removed.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskRemovedEvent {
    /// The unique identifier of the removed task.
    /// This is typically a longer, more descriptive ID for the task.
    pub task_id: String,

    /// The small ID of the removed task within the Atoma network.
    /// This ID is used for efficient referencing of the task in various operations and events.
    pub task_small_id: TaskSmallId,

    /// The epoch at which the task was permanently removed from the network.
    /// An epoch represents a specific point in time or a block height in the network.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub removed_at_epoch: u64,
}

/// Represents an event that is emitted when a new stack is created in the Atoma network.
///
/// This event contains information about the newly created stack, including its identifiers,
/// the selected node, computational resources, and pricing details.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StackCreatedEvent {
    /// The address of the owner of the stack.
    pub owner: String,

    /// The unique identifier of the created stack.
    /// This is typically a longer, more descriptive ID for the stack.
    pub stack_id: String,

    /// The small ID assigned to the stack within the Atoma network.
    /// This ID is used for efficient referencing of the stack in various operations and events.
    pub stack_small_id: StackSmallId,

    /// The small ID of the task that the stack is associated with.
    /// This is a compact identifier for the task within the Atoma network.
    pub task_small_id: TaskSmallId,

    /// The small ID of the node selected to process this stack.
    /// This identifies which node in the network is responsible for executing the stack's tasks.
    pub selected_node_id: NodeSmallId,

    /// The number of compute units allocated for this stack.
    /// This represents the computational resources reserved for processing the stack's tasks.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub num_compute_units: u64,

    /// The price associated with this stack.
    /// This value represents the cost in the network's native currency for processing this stack.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub price: u64,
}

impl From<(StackCreatedEvent, i64)> for StackCreateAndUpdateEvent {
    fn from((event, already_computed_units): (StackCreatedEvent, i64)) -> Self {
        Self {
            stack_id: event.stack_id,
            stack_small_id: event.stack_small_id,
            owner: event.owner,
            task_small_id: event.task_small_id,
            selected_node_id: event.selected_node_id,
            num_compute_units: event.num_compute_units,
            price: event.price,
            already_computed_units,
        }
    }
}

/// An event emitted when a stack is created and updated simultaneously with already computed compute units.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StackCreateAndUpdateEvent {
    /// The address of the owner of the stack.
    pub owner: String,

    /// The unique identifier of the created stack.
    /// This is typically a longer, more descriptive ID for the stack.
    pub stack_id: String,

    /// The small ID assigned to the stack within the Atoma network.
    /// This ID is used for efficient referencing of the stack in various operations and events.
    pub stack_small_id: StackSmallId,

    /// The small ID of the task that the stack is associated with.
    /// This is a compact identifier for the task within the Atoma network.
    pub task_small_id: TaskSmallId,

    /// The small ID of the node selected to process this stack.
    /// This identifies which node in the network is responsible for executing the stack's tasks.
    pub selected_node_id: NodeSmallId,

    /// The number of compute units allocated for this stack.
    /// This represents the computational resources reserved for processing the stack's tasks.
    pub num_compute_units: u64,

    /// The price associated with this stack.
    /// This value represents the cost in the network's native currency for processing this stack.
    pub price: u64,

    /// The number of compute units already computed for this stack.
    pub already_computed_units: i64,
}

/// Represents an event that is emitted when an attempt is made to settle a stack in the Atoma network.
///
/// This event contains information about the settlement attempt, including the stack and node identifiers,
/// requested attestation nodes, proofs, and claimed compute units.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StackTrySettleEvent {
    /// The small ID of the stack being settled.
    /// This is used for efficient referencing of the stack within the Atoma network.
    pub stack_small_id: StackSmallId,

    /// The small ID of the node that was selected to process this stack.
    /// This identifies which node in the network is responsible for the stack's execution.
    pub selected_node_id: NodeSmallId,

    /// A list of small IDs of nodes requested to attest to the stack's execution.
    /// These nodes are responsible for verifying the correctness of the stack's execution.
    pub requested_attestation_nodes: Vec<NodeSmallId>,

    /// The committed proof of the stack's execution.
    /// This is typically a cryptographic proof that demonstrates the correctness of the execution.
    pub committed_stack_proof: Vec<u8>,

    /// The Merkle leaf representing the stack in the network's Merkle tree.
    /// This is used for efficient verification and auditing of the stack's state.
    pub stack_merkle_leaf: Vec<u8>,

    /// The number of compute units claimed by the selected node for processing this stack.
    /// This represents the computational resources used in executing the stack's tasks.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub num_claimed_compute_units: u64,
}

/// Represents an event that is emitted when a new attestation is made for a stack settlement in the Atoma network.
///
/// This event contains information about the stack being attested, the nodes involved in the attestation process,
/// and the computational resources claimed.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NewStackSettlementAttestationEvent {
    /// The small ID of the stack being attested.
    /// This is used for efficient referencing of the stack within the Atoma network.
    pub stack_small_id: StackSmallId,

    /// The small ID of the node that made the attestation.
    /// This identifies which node in the network made the attestation.
    pub attestation_node_id: NodeSmallId,

    /// The committed proof of the stack's execution.
    /// This is typically a cryptographic proof that demonstrates the correctness of the execution.
    pub committed_stack_proof: Vec<u8>,

    /// The Merkle leaf representing the stack in the network's Merkle tree.
    /// This is used for efficient verification and auditing of the stack's state.
    pub stack_merkle_leaf: Vec<u8>,
}

/// Represents an event that is emitted when a settlement ticket is issued for a stack in the Atoma network.
///
/// This event contains information about the settled stack, including identifiers, computational claims,
/// attestation nodes, dispute resolution details, and the committed proof.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StackSettlementTicketEvent {
    /// The small ID of the stack for which the settlement ticket is issued.
    /// This is used for efficient referencing of the stack within the Atoma network.
    pub stack_small_id: StackSmallId,

    /// The small ID of the node that was selected to process this stack.
    /// This identifies which node in the network was responsible for executing the stack's tasks.
    pub selected_node_id: NodeSmallId,

    /// The number of compute units claimed by the selected node for processing this stack.
    /// This represents the computational resources used in executing the stack's tasks.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub num_claimed_compute_units: u64,

    /// A list of small IDs of nodes that were requested to attest to the stack's execution.
    /// These nodes are responsible for verifying the correctness of the stack's execution.
    pub requested_attestation_nodes: Vec<NodeSmallId>,

    /// The epoch at which any disputes related to this stack settlement were resolved.
    /// An epoch represents a specific point in time or a block height in the network.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub dispute_settled_at_epoch: u64,

    /// The committed proof of the stack's execution.
    /// This is typically a cryptographic proof that demonstrates the correctness of the execution.
    pub committed_stack_proof: Vec<u8>,
}

/// Represents an event that is emitted when a stack settlement ticket is claimed in the Atoma network.
///
/// This event contains information about the claimed settlement, including stack and node identifiers,
/// attestation nodes, computational claims, and any refund amount for the user.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StackSettlementTicketClaimedEvent {
    /// The small ID of the stack for which the settlement ticket was claimed.
    /// This is used for efficient referencing of the stack within the Atoma network.
    pub stack_small_id: StackSmallId,

    /// The small ID of the node that was selected to process this stack.
    /// This identifies which node in the network was responsible for executing the stack's tasks.
    pub selected_node_id: NodeSmallId,

    /// A list of small IDs of nodes that attested to the stack's execution.
    /// These nodes were responsible for verifying the correctness of the stack's execution.
    pub attestation_nodes: Vec<NodeSmallId>,

    /// The number of compute units claimed by the selected node for processing this stack.
    /// This represents the computational resources used in executing the stack's tasks.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub num_claimed_compute_units: u64,

    /// The amount of refund, if any, issued to the user for this stack settlement.
    /// This is represented as a vector of bytes, likely to accommodate different currency representations.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub user_refund_amount: u64,
}

/// Represents an event that is emitted when there's a dispute in the attestation process for a stack in the Atoma network.
///
/// This event contains information about the disputed stack, the attestation commitment, and the original execution commitment.
/// It's used to track and resolve conflicts between the attesting node and the original executing node.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StackAttestationDisputeEvent {
    /// The small ID of the stack for which the attestation is disputed.
    /// This is used for efficient referencing of the stack within the Atoma network.
    pub stack_small_id: StackSmallId,

    /// The commitment provided by the attestation node.
    /// This is typically a cryptographic commitment that represents the attestation node's view of the stack execution.
    pub attestation_commitment: Vec<u8>,

    /// The small ID of the node that provided the disputed attestation.
    /// This identifies which node in the network is challenging the original execution.
    pub attestation_node_id: NodeSmallId,

    /// The small ID of the node that originally executed the stack.
    /// This identifies which node in the network was responsible for the initial execution of the stack's tasks.
    pub original_node_id: NodeSmallId,

    /// The original commitment provided by the executing node.
    /// This is typically a cryptographic commitment that represents the original node's claim about the stack execution.
    pub original_commitment: Vec<u8>,
}

/// Represents the parameters for a text-to-image prompt.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Text2ImagePromptParams {
    /// The guidance scale for the diffusion process, represented as a floating point number stored as little-endian bytes.
    /// This value controls how closely the image adheres to the prompt.
    pub guidance_scale: u32,

    /// The height of the generated image in pixels.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub height: u64,

    /// Optional input image for image-to-image generation, stored as bytes.
    /// If provided, the model will use this image as a starting point for the generation process.
    pub img2img: Option<Vec<u8>>,

    /// The strength of the image-to-image transformation, represented as a floating point number stored as little-endian bytes.
    /// This value determines how much the input image influences the final output.
    pub img2img_strength: u32,

    /// The name or identifier of the AI model to be used for image generation.
    pub model: String,

    /// The number of denoising steps to perform during the diffusion process.
    /// More steps generally result in higher quality images but take longer to generate.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub n_steps: u64,

    /// The number of images to generate.
    /// The user pays for each image generated.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub num_samples: u64,

    /// The destination where the generated image(s) will be stored, represented as a byte vector.
    pub output_destination: Vec<u8>,

    /// The text prompt describing the desired image, stored as a byte vector (typically UTF-8 encoded).
    pub prompt: Vec<u8>,

    /// A seed value for the random number generator used in image generation.
    /// Using the same seed with the same input will produce deterministic results.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub random_seed: u64,

    /// The unconditional prompt (negative prompt) used to guide what should not appear in the image, stored as a byte vector.
    pub uncond_prompt: Vec<u8>,

    /// The width of the generated image in pixels.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub width: u64,
}

/// Represents an event emitted when a text-to-image prompt is submitted.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Text2ImagePromptEvent {
    /// The ID of the settlement object.
    pub ticket_id: String,

    /// The parameters of the prompt that nodes must evaluate.
    pub params: Text2ImagePromptParams,

    /// Determines into how many chunks the nodes split the output when
    /// they generate proof hashes.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub chunks_count: u64,

    /// The list of nodes that may be used to evaluate the prompt.
    /// Note: This might not be the final list of nodes used.
    pub nodes: Vec<NodeSmallId>,

    /// The output destination where the output will be stored.
    /// The output is serialized with MessagePack.
    pub output_destination: Vec<u8>,
}

/// Represents an event emitted when the first submission is made in a settlement process.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FirstSubmissionEvent {
    /// The ID of the settlement object.
    pub ticket_id: String,

    /// The small ID of the node that made the first submission.
    pub node_id: NodeSmallId,
}

/// Represents an event emitted when a dispute occurs during settlement.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DisputeEvent {
    /// The ID of the settlement object.
    pub ticket_id: String,

    /// The small ID of the node that made the first submission.
    pub timeout: Option<TimeoutInfo>,
}

/// Represents an event emitted when a new set of nodes is sampled for a task.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NewlySampledNodesEvent {
    /// The ID of the settlement object.
    pub ticket_id: String,

    /// The list of newly sampled nodes for the task.
    pub new_nodes: Vec<MapNodeToChunk>,
}

/// Represents an event emitted when a ticket is settled.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SettledEvent {
    /// The ID of the settlement object.   
    pub ticket_id: String,

    /// The oracle node ID that settled the ticket.
    pub oracle_node_id: Option<NodeSmallId>,
}

/// Represents an event emitted when a retry settlement is requested.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RetrySettlementEvent {
    /// The ID of the settlement object.   
    pub ticket_id: String,

    /// The number of nodes in the echelon that should be used to retry the settlement.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub how_many_nodes_in_echelon: u64,
}
/// Represents the parameters for a text-to-text prompt in the Atoma network.
///
/// This struct encapsulates all the necessary configuration options for executing
/// a text-to-text AI task, such as language generation or text completion.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Text2TextPromptParams {
    /// The maximum number of tokens to generate in the output.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub max_tokens: u64,

    /// The name or identifier of the AI model to be used for text generation.
    pub model: String,

    /// A vector of token IDs to be prepended to the input prompt.
    /// These tokens can be used to provide context or control the model's behavior.
    pub pre_prompt_tokens: Vec<u32>,

    /// If true, the generated output will be prepended with the input prompt.
    /// This can be useful for maintaining context in the output.
    pub prepend_output_with_input: bool,

    /// The input prompt as a byte vector. This is typically UTF-8 encoded text.
    pub prompt: Vec<u8>,

    /// A seed value for the random number generator used in text generation.
    /// Using the same seed with the same input will produce deterministic results.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub random_seed: u64,

    /// The number of previous tokens to consider when applying the repeat penalty.
    /// This helps prevent the model from repeating the same phrases too frequently.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub repeat_last_n: u64,

    /// The penalty applied to repeated tokens, stored as a 32-bit float in little-endian byte order.
    /// Higher values make the model less likely to repeat itself.
    pub repeat_penalty: u32,

    /// If true, the output will be streamed token by token instead of returned all at once.
    /// This can be useful for real-time applications or long-form text generation.
    pub should_stream_output: bool,

    /// The temperature parameter for controlling randomness in token selection,
    /// stored as a 32-bit float in little-endian byte order.
    /// Higher values make the output more random, while lower values make it more deterministic.
    pub temperature: u32,

    /// The number of highest probability vocabulary tokens to keep for top-k filtering.
    /// This helps control the diversity of the generated text.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub top_k: u64,

    /// The cumulative probability threshold for top-p (nucleus) filtering,
    /// stored as a 32-bit float in little-endian byte order.
    /// Only the most probable tokens with cumulative probability less than this value are considered.
    pub top_p: u32,
}

/// Represents an event emitted when a text-to-text prompt is submitted.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Text2TextPromptEvent {
    /// The ID of the settlement object.
    pub ticket_id: String,

    /// The parameters of the prompt that nodes must evaluate.
    pub params: Text2TextPromptParams,

    /// Determines into how many chunks the nodes split the output when
    /// they generate proof hashes.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub chunks_count: u64,

    /// The list of nodes that may be used to evaluate the prompt.
    /// Note: This might not be the final list of nodes used.
    pub nodes: Vec<NodeSmallId>,

    /// The output destination where the output will be stored.
    /// The output is serialized with MessagePack.
    pub output_destination: Vec<u8>,
}

/// Represents an event emitted when a node's public key is rotated.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodePublicKeyCommittmentEvent {
    /// The epoch number when the node key rotation was requested.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub epoch: u64,

    /// The counter for the number of times the contract has requested nodes rotating their public keys.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub key_rotation_counter: u64,

    /// The small ID of the node that requested the key rotation.
    pub node_id: NodeSmallId,

    /// The node's new registered public key.
    pub new_public_key: Vec<u8>,

    /// The TEE remote attestation report attesting for
    /// the public key's generation integrity, in byte format.
    pub tee_remote_attestation_bytes: Vec<u8>,
}

/// Represents an event emitted when Atoma's smart contract requests new node key rotation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NewKeyRotationEvent {
    /// The epoch number when the node key rotation was requested.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub epoch: u64,

    /// The counter for the number of times the contract has requested nodes rotating their public keys.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub key_rotation_counter: u64,
}

/// Represents an identifier for an echelon (performance tier) in the Atoma network.
///
/// Echelons are used to categorize nodes based on their performance capabilities
/// or the level of service they can provide.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EchelonId {
    /// The unique numerical identifier for the echelon.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub id: u64,
}

/// Represents a compact identifier for a node in the Atoma network.
///
/// This small ID is used for efficient referencing of nodes in various
/// operations and events within the network.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeSmallId {
    /// The unique numerical identifier for the node.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub inner: u64,
}

/// Represents a compact identifier for a stack in the Atoma network.
///
/// A stack typically refers to a collection of tasks or operations that
/// need to be processed together. This small ID allows for efficient
/// referencing of stacks in various operations and events.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StackSmallId {
    /// The unique numerical identifier for the stack.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub inner: u64,
}

/// Represents a compact identifier for a task in the Atoma network.
///
/// Tasks are units of work that need to be performed by nodes in the network.
/// This small ID allows for efficient referencing of tasks in various
/// operations and events.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskSmallId {
    /// The unique numerical identifier for the task.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub inner: u64,
}

/// Represents the role of a task in the Atoma network.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskRole {
    /// The unique numerical identifier for the task role.
    pub inner: u16,
}

/// Represents the security level of a task in the Atoma network.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SecurityLevel {
    /// The unique numerical identifier for the security level.
    /// Possible values are:
    /// - 0: No security
    /// - 1: Sampling Consensus
    /// - 2: Confidential compute (through trusted hardware)
    pub inner: u16,
}

/// Represents information about a timeout in the Atoma network.
///
/// This struct contains details about the timeout, such as the number of times
/// the timeout has occurred, the timeout duration in milliseconds, and the epoch
/// and timestamp when the timeout started.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TimeoutInfo {
    /// How many times has the settlement timed out.
    /// Once this reaches a threshold `MaxTicketTimeouts`, the ticket
    /// will be disputed.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub timed_out_count: u64,

    /// If the settlement takes more than this, the settlement can be cut
    /// short.
    /// See the `try_to_settle` endpoint.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub timeout_ms: u64,

    /// Will be relevant for timeouting.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub started_in_epoch: u64,

    /// Will be relevant for timeouting.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub started_at_epoch_timestamp_ms: u64,
}

/// Represents a mapping between a node and a chunk in the Atoma network.
///
/// This struct is used to associate a specific node that needs to attest the correctness of some submitted output,
/// typically in the context of settlement of responses. We need to make sure that the order of chunks is correct,
/// so we can reassemble the output in the correct order.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MapNodeToChunk {
    /// The small ID of the node assigned to process this chunk.
    /// This identifies which node in the network is responsible for this specific portion of work.
    pub node_id: NodeSmallId,

    /// The order or position of this chunk within the overall task or dataset.
    /// This helps maintain the correct sequence when processing or reassembling distributed work.
    #[serde(deserialize_with = "deserialize_string_to_u64")]
    pub order: u64,
}

#[derive(Debug, Error)]
pub enum SuiEventParseError {
    #[error("Unknown event error: {0}")]
    UnknownEvent(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_published_event_deserialization() {
        let json = json!({
        "db": "0x123",
        "manager_badge": "0x456"
        });
        let event: PublishedEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.db, "0x123");
        assert_eq!(event.manager_badge, "0x456");
    }

    #[test]
    fn test_node_registered_event_deserialization() {
        let json = json!({
            "badge_id": "0x789",
            "node_small_id": {"inner": "42"}
        });
        let event: NodeRegisteredEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.badge_id, "0x789");
        assert_eq!(event.node_small_id.inner, 42);
    }

    #[test]
    fn test_node_subscribed_to_model_event_deserialization() {
        let json = json!({
            "node_small_id": {"inner": "1"},
            "model_name": "gpt-3",
            "echelon_id": {"id": "2"}
        });
        let event: NodeSubscribedToModelEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.node_small_id.inner, 1);
        assert_eq!(event.model_name, "gpt-3");
        assert_eq!(event.echelon_id.id, 2);
    }

    #[test]
    fn test_node_subscribed_to_task_event_deserialization() {
        let json = json!({
            "task_small_id": {"inner": "3"},
            "node_small_id": {"inner": "4"},
            "price_per_one_million_compute_units": "100",
            "max_num_compute_units": "1000"
        });
        let event: NodeSubscribedToTaskEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.task_small_id.inner, 3);
        assert_eq!(event.node_small_id.inner, 4);
        assert_eq!(event.price_per_one_million_compute_units, 100);
        assert_eq!(event.max_num_compute_units, 1000);
    }

    #[test]
    fn test_node_subscription_updated_event_deserialization() {
        let json = json!({
            "task_small_id": {"inner": "3"},
            "node_small_id": {"inner": "4"},
            "price_per_one_million_compute_units": "150",
            "max_num_compute_units": "1500"
        });
        let event: NodeSubscriptionUpdatedEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.task_small_id.inner, 3);
        assert_eq!(event.node_small_id.inner, 4);
        assert_eq!(event.price_per_one_million_compute_units, 150);
        assert_eq!(event.max_num_compute_units, 1500);
    }

    #[test]
    fn test_node_unsubscribed_from_task_event_deserialization() {
        let json = json!({
            "task_small_id": {"inner": "5"},
            "node_small_id": {"inner": "6"}
        });
        let event: NodeUnsubscribedFromTaskEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.task_small_id.inner, 5);
        assert_eq!(event.node_small_id.inner, 6);
    }

    #[test]
    fn test_task_registered_event_deserialization() {
        let json = json!({
            "task_id": "task-001",
            "task_small_id": {"inner": "7"},
            "role": {"inner": 1},
            "model_name": "gpt-3",
            "security_level": {"inner": 2},
            "minimum_reputation_score": 80
        });
        let event: TaskRegisteredEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.task_id, "task-001");
        assert_eq!(event.task_small_id.inner, 7);
        assert_eq!(event.role.inner, 1);
        assert_eq!(event.model_name, Some("gpt-3".to_string()));
        assert_eq!(event.security_level.inner, 2);
        assert_eq!(event.minimum_reputation_score, Some(80));
    }

    #[test]
    fn test_task_deprecation_event_deserialization() {
        let json = json!({
            "task_id": "task-002",
            "task_small_id": {"inner": "8"},
            "epoch": "1000"
        });
        let event: TaskDeprecationEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.task_id, "task-002");
        assert_eq!(event.task_small_id.inner, 8);
        assert_eq!(event.epoch, 1000);
    }

    #[test]
    fn test_task_removed_event_deserialization() {
        let json = json!({
            "task_id": "task-003",
            "task_small_id": {"inner": "9"},
            "removed_at_epoch": "2000"
        });
        let event: TaskRemovedEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.task_id, "task-003");
        assert_eq!(event.task_small_id.inner, 9);
        assert_eq!(event.removed_at_epoch, 2000);
    }

    #[test]
    fn test_stack_created_event_deserialization() {
        let json = json!({
            "owner": "0x123",
            "stack_id": "stack-001",
            "stack_small_id": {"inner": "10"},
            "task_small_id": {"inner": "3"},
            "selected_node_id": {"inner": "11"},
            "num_compute_units": "5",
            "price": "1000"
        });
        let event: StackCreatedEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.owner, "0x123");
        assert_eq!(event.stack_id, "stack-001");
        assert_eq!(event.stack_small_id.inner, 10);
        assert_eq!(event.task_small_id.inner, 3);
        assert_eq!(event.selected_node_id.inner, 11);
        assert_eq!(event.num_compute_units, 5);
        assert_eq!(event.price, 1000);
    }

    #[test]
    fn test_stack_try_settle_event_deserialization() {
        let json = json!({
            "stack_small_id": {"inner": "12"},
            "selected_node_id": {"inner": "13"},
            "requested_attestation_nodes": [{"inner": "14"}, {"inner": "15"}],
            "committed_stack_proof": [1, 2, 3],
            "stack_merkle_leaf": [4, 5, 6],
            "num_claimed_compute_units": "100"
        });
        let event: StackTrySettleEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.stack_small_id.inner, 12);
        assert_eq!(event.selected_node_id.inner, 13);
        assert_eq!(event.requested_attestation_nodes.len(), 2);
        assert_eq!(event.requested_attestation_nodes[0].inner, 14);
        assert_eq!(event.requested_attestation_nodes[1].inner, 15);
        assert_eq!(event.committed_stack_proof, vec![1, 2, 3]);
        assert_eq!(event.stack_merkle_leaf, vec![4, 5, 6]);
        assert_eq!(event.num_claimed_compute_units, 100);
    }

    #[test]
    fn test_new_stack_settlement_attestation_event_deserialization() {
        let json = json!({
            "stack_small_id": {"inner": "16"},
            "attestation_node_id": {"inner": "17"},
            "committed_stack_proof": [1, 2, 3],
            "stack_merkle_leaf": [4, 5, 6]
        });
        let event: NewStackSettlementAttestationEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.stack_small_id.inner, 16);
        assert_eq!(event.attestation_node_id.inner, 17);
        assert_eq!(event.committed_stack_proof, vec![1, 2, 3]);
        assert_eq!(event.stack_merkle_leaf, vec![4, 5, 6]);
    }

    #[test]
    fn test_stack_settlement_ticket_event_deserialization() {
        let json = json!({
            "stack_small_id": {"inner": "19"},
            "selected_node_id": {"inner": "20"},
            "num_claimed_compute_units": "300",
            "requested_attestation_nodes": [{"inner": "21"}, {"inner": "22"}],
            "dispute_settled_at_epoch": "3000",
            "committed_stack_proof": [7, 8, 9]
        });
        let event: StackSettlementTicketEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.stack_small_id.inner, 19);
        assert_eq!(event.selected_node_id.inner, 20);
        assert_eq!(event.num_claimed_compute_units, 300);
        assert_eq!(event.requested_attestation_nodes.len(), 2);
        assert_eq!(event.requested_attestation_nodes[0].inner, 21);
        assert_eq!(event.requested_attestation_nodes[1].inner, 22);
        assert_eq!(event.dispute_settled_at_epoch, 3000);
        assert_eq!(event.committed_stack_proof, vec![7, 8, 9]);
    }

    #[test]
    fn test_stack_settlement_ticket_claimed_event_deserialization() {
        let json = json!({
            "stack_small_id": {"inner": "23"},
            "selected_node_id": {"inner": "24"},
            "attestation_nodes": [{"inner": "25"}, {"inner": "26"}],
            "num_claimed_compute_units": "400",
            "user_refund_amount": "100"
        });
        let event: StackSettlementTicketClaimedEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.stack_small_id.inner, 23);
        assert_eq!(event.selected_node_id.inner, 24);
        assert_eq!(event.attestation_nodes.len(), 2);
        assert_eq!(event.attestation_nodes[0].inner, 25);
        assert_eq!(event.attestation_nodes[1].inner, 26);
        assert_eq!(event.num_claimed_compute_units, 400);
        assert_eq!(event.user_refund_amount, 100);
    }

    #[test]
    fn test_stack_attestation_dispute_event_deserialization() {
        let json = json!({
            "stack_small_id": {"inner": "27"},
            "attestation_commitment": [13, 14, 15],
            "attestation_node_id": {"inner": "28"},
            "original_node_id": {"inner": "29"},
            "original_commitment": [16, 17, 18]
        });
        let event: StackAttestationDisputeEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.stack_small_id.inner, 27);
        assert_eq!(event.attestation_commitment, vec![13, 14, 15]);
        assert_eq!(event.attestation_node_id.inner, 28);
        assert_eq!(event.original_node_id.inner, 29);
        assert_eq!(event.original_commitment, vec![16, 17, 18]);
    }

    #[test]
    fn test_first_submission_event_deserialization() {
        let json = json!({
            "ticket_id": "ticket-003",
            "node_id": {"inner": "30"}
        });
        let event: FirstSubmissionEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.ticket_id, "ticket-003");
        assert_eq!(event.node_id.inner, 30);
    }

    #[test]
    fn test_dispute_event_deserialization() {
        let json = json!({
            "ticket_id": "ticket-004",
            "timeout": {
                "timed_out_count": "2",
                "timeout_ms": "5000",
                "started_in_epoch": "4000",
                "started_at_epoch_timestamp_ms": "162000"
            }
        });
        let event: DisputeEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.ticket_id, "ticket-004");
        assert!(event.timeout.is_some());
        let timeout = event.timeout.unwrap();
        assert_eq!(timeout.timed_out_count, 2);
        assert_eq!(timeout.timeout_ms, 5000);
        assert_eq!(timeout.started_in_epoch, 4000);
        assert_eq!(timeout.started_at_epoch_timestamp_ms, 162000);
    }

    #[test]
    fn test_newly_sampled_nodes_event_deserialization() {
        let json = json!({
            "ticket_id": "ticket-005",
            "new_nodes": [
                {"node_id": {"inner": "31"}, "order": "0"},
                {"node_id": {"inner": "32"}, "order": "1"}
            ]
        });
        let event: NewlySampledNodesEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.ticket_id, "ticket-005");
        assert_eq!(event.new_nodes.len(), 2);
        assert_eq!(event.new_nodes[0].node_id.inner, 31);
        assert_eq!(event.new_nodes[0].order, 0);
        assert_eq!(event.new_nodes[1].node_id.inner, 32);
        assert_eq!(event.new_nodes[1].order, 1);
    }

    #[test]
    fn test_settled_event_deserialization() {
        let json = json!({
            "ticket_id": "ticket-006",
            "oracle_node_id": {"inner": "33"}
        });
        let event: SettledEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.ticket_id, "ticket-006");
        assert!(event.oracle_node_id.is_some());
        assert_eq!(event.oracle_node_id.unwrap().inner, 33);
    }

    #[test]
    fn test_retry_settlement_event_deserialization() {
        let json = json!({
            "ticket_id": "ticket-007",
            "how_many_nodes_in_echelon": "5"
        });
        let event: RetrySettlementEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.ticket_id, "ticket-007");
        assert_eq!(event.how_many_nodes_in_echelon, 5);
    }

    #[test]
    fn test_text2image_prompt_event_deserialization() {
        let json = json!({
            "ticket_id": "ticket-001",
            "params": {
                "guidance_scale": 7,
                "height": "512",
                "img2img": null,
                "img2img_strength": 0,
                "model": "stable-diffusion-v1-5",
                "n_steps": "50",
                "num_samples": "1",
                "output_destination": [1, 2, 3],
                "prompt": [65, 66, 67],
                "random_seed": "42",
                "uncond_prompt": [68, 69, 70],
                "width": "512"
            },
            "chunks_count": "4",
            "nodes": [{"inner": "1"}, {"inner": "2"}],
            "output_destination": [4, 5, 6]
        });
        let event: Text2ImagePromptEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.ticket_id, "ticket-001");
        assert_eq!(event.params.guidance_scale, 7);
        assert_eq!(event.params.height, 512);
        assert_eq!(event.params.model, "stable-diffusion-v1-5");
        assert_eq!(event.chunks_count, 4);
        assert_eq!(event.nodes.len(), 2);
        assert_eq!(event.nodes[0].inner, 1);
        assert_eq!(event.nodes[1].inner, 2);
        assert_eq!(event.output_destination, vec![4, 5, 6]);
    }

    #[test]
    fn test_text2text_prompt_event_deserialization() {
        let json = json!({
            "ticket_id": "ticket-002",
            "params": {
                "max_tokens": "100",
                "model": "gpt-3",
                "pre_prompt_tokens": [1, 2, 3],
                "prepend_output_with_input": true,
                "prompt": [65, 66, 67],
                "random_seed": "42",
                "repeat_last_n": "64",
                "repeat_penalty": 1065353216,  // 1.0 in IEEE 754 single-precision float
                "should_stream_output": false,
                "temperature": 1065353216,  // 1.0 in IEEE 754 single-precision float
                "top_k": "50",
                "top_p": 1065353216  // 1.0 in IEEE 754 single-precision float
            },
            "chunks_count": "2",
            "nodes": [{"inner": "3"}, {"inner": "4"}],
            "output_destination": [7, 8, 9]
        });
        let event: Text2TextPromptEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.ticket_id, "ticket-002");
        assert_eq!(event.params.max_tokens, 100);
        assert_eq!(event.params.model, "gpt-3");
        assert_eq!(event.params.pre_prompt_tokens, vec![1, 2, 3]);
        assert!(event.params.prepend_output_with_input);
        assert_eq!(event.params.prompt, vec![65, 66, 67]);
        assert_eq!(event.params.random_seed, 42);
        assert_eq!(event.params.repeat_last_n, 64);
        assert_eq!(event.params.repeat_penalty, 1065353216);
        assert!(!event.params.should_stream_output);
        assert_eq!(event.params.temperature, 1065353216);
        assert_eq!(event.params.top_k, 50);
        assert_eq!(event.params.top_p, 1065353216);
        assert_eq!(event.chunks_count, 2);
        assert_eq!(event.nodes.len(), 2);
        assert_eq!(event.nodes[0].inner, 3);
        assert_eq!(event.nodes[1].inner, 4);
        assert_eq!(event.output_destination, vec![7, 8, 9]);
    }
}
