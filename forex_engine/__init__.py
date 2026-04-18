"""Forex Context Engine — public surface.

The package is intentionally flat: every node, the orchestrator, the
Gen AI client factory, the config, the models, and the time policy are
all reachable via ``forex_engine.<name>``.
"""
from .config import EngineConfig, load_from_env
from .exceptions import (
    ConfigurationError,
    DisproofError,
    EvolutionError,
    ExtractionError,
    ForexEngineError,
    GenAIBackendError,
    PersistenceError,
    SchemaValidationError,
    ValidationError,
)
from .genai_backend import (
    AnthropicClient,
    EchoClient,
    GenAIClient,
    OpenAIClient,
    build_client,
    compute_cost,
)
from .logging_setup import audit, configure_logging, reject
from .models import (
    Confidence,
    Conflict,
    ContextState,
    DisproofResult,
    Event,
    Hypothesis,
    InferenceRequest,
    InferenceResponse,
    Relationship,
    Severity,
    Signal,
    SignalKind,
    SignalSource,
    StateDelta,
    ValidationResult,
)
from .node_a_extraction import Extractor
from .node_b_evolution import Evolver, EvolutionResult
from .node_c_validation import Validator
from .node_d_persistence import (
    InMemoryStateRepository,
    PostgresStateRepository,
    StateRepository,
)
from .node_e_disproof import DisproofEngine
from .orchestrator import (
    Orchestrator,
    PipelineResult,
    RawPayload,
    build_default_orchestrator,
)
from .time_utils import EST, now_est, require_est

__all__ = [
    # config / errors
    "EngineConfig", "load_from_env",
    "ForexEngineError", "ConfigurationError", "SchemaValidationError",
    "ExtractionError", "EvolutionError", "PersistenceError",
    "ValidationError", "DisproofError", "GenAIBackendError",
    # time
    "EST", "now_est", "require_est",
    # logging
    "configure_logging", "audit", "reject",
    # models
    "Signal", "SignalKind", "SignalSource", "Confidence",
    "Event", "Relationship", "ContextState", "StateDelta",
    "ValidationResult", "Conflict", "Severity",
    "Hypothesis", "DisproofResult",
    "InferenceRequest", "InferenceResponse",
    # nodes
    "Extractor", "Evolver", "EvolutionResult",
    "Validator", "DisproofEngine",
    "StateRepository", "InMemoryStateRepository", "PostgresStateRepository",
    # genai
    "GenAIClient", "AnthropicClient", "OpenAIClient", "EchoClient",
    "build_client", "compute_cost",
    # orchestration
    "Orchestrator", "PipelineResult", "RawPayload",
    "build_default_orchestrator",
]
