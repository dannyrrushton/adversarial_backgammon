"""Human interface: teacher, adversary and AI-vs-AI sessions with text and 3D front ends."""

from .factory import Options, build_session
from .session import Event, GameSession, Mode, Seat, SessionConfig

__all__ = ["Event", "GameSession", "Mode", "Options", "Seat", "SessionConfig", "build_session"]
