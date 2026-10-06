"""Exceptions shared across the pipeline."""


class Cancelled(Exception):
    """Raised inside a worker when the user asks to stop."""
