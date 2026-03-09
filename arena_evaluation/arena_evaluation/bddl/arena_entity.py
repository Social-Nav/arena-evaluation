"""Arena entity wrapper for BDDL scope objects.

Each ArenaEntity represents a named object in the simulation (robot, obstacle,
landmark, room) whose state can be updated in-place every tick.  The ``get_*``
methods are called by the BDDL predicate classes during ``evaluate()``.

The design follows the same contract as ``TrivialGenericObject`` in the
upstream ``bddl`` package: each object exposes per-predicate query methods
that the corresponding ``AtomicFormula._evaluate()`` dispatches to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ArenaEntity:
    """Mutable entity whose pose is updated by the evaluator each tick."""

    name: str
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    radius: float = 0.3  # bounding circle for collision / touching checks

    # For room-type entities
    is_room: bool = False
    room_radius: float = 0.0  # only meaningful when is_room=True

    # --- Predicate helpers (called by ArenaBackend predicate classes) ---

    def get_nextto(self, other: "ArenaEntity") -> bool:
        """True when Euclidean distance ≤ sum of radii + threshold (default 0.5 m)."""
        return _dist(self, other) <= (self.radius + other.radius + 0.5)

    def get_inroom(self, room: "ArenaEntity") -> bool:
        """True when this entity is inside *room*'s circular boundary."""
        if not room.is_room:
            return False
        return _dist(self, room) <= room.room_radius

    def get_touching(self, other: "ArenaEntity") -> bool:
        """True when bounding circles overlap (distance ≤ sum of radii)."""
        return _dist(self, other) <= (self.radius + other.radius)

    # --- Convenience ---

    def update_pose(self, x: float, y: float, yaw: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw


def _dist(a: ArenaEntity, b: ArenaEntity) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)
