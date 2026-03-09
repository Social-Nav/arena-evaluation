"""Arena-specific BDDL configuration.

Patches ``bddl.config.get_domain_filename`` so that the domain name ``"arena"``
resolves to our bundled ``domain_arena.bddl`` instead of looking inside the
upstream ``bddl`` package's ``activity_definitions/`` directory.

Also provides ``parse_arena_problem()`` which wraps ``bddl.parsing.parse_problem``
with the ``predefined_problem`` parameter for inline BDDL strings.
"""

from __future__ import annotations

import os
import pathlib

import bddl.config
import bddl.parsing

_DEFINITIONS_DIR = pathlib.Path(__file__).resolve().parent / "definitions"
_ARENA_DOMAIN_FILE = _DEFINITIONS_DIR / "domain_arena.bddl"

# Keep the original so we can fall back
_original_get_domain_filename = bddl.config.get_domain_filename


def _patched_get_domain_filename(domain_name: str) -> str:
    if domain_name == "arena":
        return str(_ARENA_DOMAIN_FILE)
    return _original_get_domain_filename(domain_name)


def install() -> None:
    """Monkey-patch bddl.config so ``parse_domain("arena")`` works.

    We must patch both ``bddl.config`` (the source) and ``bddl.parsing``
    (which imported ``get_domain_filename`` at import time via ``from``).
    """
    bddl.config.get_domain_filename = _patched_get_domain_filename
    bddl.parsing.get_domain_filename = _patched_get_domain_filename


def parse_arena_problem(problem_bddl: str):
    """Parse an inline BDDL problem string against the arena domain.

    Returns:
        (problem_name, parsed_objects, parsed_initial_conditions, parsed_goal_conditions)
    """
    install()  # ensure our domain is registered
    domain_name, *_ = bddl.parsing.parse_domain("arena")
    return bddl.parsing.parse_problem(
        behavior_activity="",
        activity_definition=0,
        domain_name=domain_name,
        predefined_problem=problem_bddl,
    )
