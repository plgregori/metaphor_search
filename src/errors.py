class ConfigError(ValueError):
    """Models configuration file missing, unreadable or invalid."""

class DuplicateModelError(ValueError):
    """The alias is already registered."""

class UnknownModelError(ValueError):
    """The alias is not registered."""

class DeleteBlockedError(ValueError):
    """The model cannot be deleted right now."""

class OllamaUnavailableError(ConnectionError):
    """Ollama is unreachable, too slow, or refused the request."""

class PromptFileError(ValueError):
    """A prompts CSV file is missing, unreadable or invalid."""

class InvalidRunRequestError(ValueError):
    """The run request cannot be executed as asked."""

class RunStateError(ValueError):
    """The operation is not possible in the run's current state (another run is active, or this one is not running)."""

class UnknownRunError(ValueError):
    """The run id does not exist."""