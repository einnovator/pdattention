"""Registry for model-level position-encoding strategies."""

from .absolute import AbsolutePositionEncoding
from .rope import RotaryPositionEncoding
from .sinusoidal import SinusoidalPositionEncoding


def build_position_encoding(mode: str, *, head_dim: int, rope_theta: float = 10_000.0):
    """Construct a validated position strategy from configuration values."""
    normalized = str(mode).lower()
    if normalized == "absolute":
        return AbsolutePositionEncoding()
    if normalized == "rope":
        return RotaryPositionEncoding(head_dim=head_dim, theta=rope_theta)
    if normalized == "sinusoidal":
        return SinusoidalPositionEncoding()
    raise ValueError(f"Unsupported position_encoding: {mode}")
