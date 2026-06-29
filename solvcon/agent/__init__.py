# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Agent: drive the 2D drawing world with an AI backend, with or without GUI.

This package lives outside :mod:`solvcon.pilot` so it can also drive pure
computation that needs no graphics.  The headless core (:mod:`_core`) and the
backend abstraction (:mod:`_backend`) load without Qt, so they run in CI and a
headless build.
"""

from . import _core  # noqa: F401
from . import _backend  # noqa: F401

AgentSession = _core.AgentSession
TranscriptTurn = _core.TranscriptTurn
AgentBackend = _backend.AgentBackend
BackendResponse = _backend.BackendResponse
EchoBackend = _backend.EchoBackend
register = _backend.register
all_backends = _backend.all_backends
available_backends = _backend.available_backends
get_backend = _backend.get_backend

# TODO: when the Qt dock module exists in solvcon.pilot, point this at its
# Agent class, guarded by _pilot_core.enable like the airfoil sub-package.
Agent = None

__all__ = [  # noqa: F822
    'AgentSession',
    'TranscriptTurn',
    'AgentBackend',
    'BackendResponse',
    'EchoBackend',
    'register',
    'all_backends',
    'available_backends',
    'get_backend',
    'Agent',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
