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