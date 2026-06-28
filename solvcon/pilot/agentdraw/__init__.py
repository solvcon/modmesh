# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Agent Draw front-end for the ``World`` API.

This package hosts the single command schema (JSON Schema, authored in
Python) that every agent front-end rides on, its validator, and the executor
that applies commands to a ``World``. The harness and MCP adapters are built
over the same schema in later work.
"""

from .schema import (  # noqa: F401
    COMMAND_SCHEMAS,
    CRUD_CATEGORIES,
    RESULT_SCHEMAS,
    SCHEMA,
    CommandError,
    apply_defaults,
    commands_by_category,
    tool_definitions,
    validate_command,
    validate_result,
    validate_script,
)
from .executor import (  # noqa: F401
    CommandResult,
    Executor,
)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
