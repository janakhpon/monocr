class MonOCRError(Exception):
    """Base exception for MonOCR errors."""
    pass

class ModelNotFoundError(MonOCRError):
    """Raised when the model file cannot be found."""
    pass

class CharsetNotFoundError(MonOCRError):
    """Raised when the charset file cannot be found."""
    pass

class ImageLoadError(MonOCRError):
    """Raised when an image cannot be loaded or processed."""
    pass
