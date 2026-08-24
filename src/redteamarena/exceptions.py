"""Typed errors used to map failures to stable CLI exit codes."""


class RedTeamArenaError(Exception):
    """Base application error."""


class ConfigurationError(RedTeamArenaError):
    """Configuration is absent or invalid."""


class ValidationError(RedTeamArenaError):
    """Generated or persisted data failed validation."""


class PipelineError(RedTeamArenaError):
    """The pipeline could not complete."""


class ProviderError(PipelineError):
    """A provider request or response failed."""


class ProviderTimeout(ProviderError):
    """A provider request timed out."""
