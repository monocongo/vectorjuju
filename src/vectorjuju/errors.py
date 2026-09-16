"""Base error shared by every vectorjuju failure a caller can catch.

Lives in its own leaf module rather than in ``convert`` so the pipeline
stages -- which ``convert()`` imports -- can raise and subclass it without
importing the module that wires them together.
"""


class VectorjujuError(Exception):
    """Base class for vectorjuju errors."""
