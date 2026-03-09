"""BDDL evaluator: evaluate episode trajectories against BDDL task definitions.

This module replays an EpisodeLog frame-by-frame and evaluates goal conditions
using the real ``bddl`` 3.x package (``bddl.condition_evaluation.compile_state``
/ ``evaluate_state``), bridged through :class:`ArenaBackend` and
:class:`ArenaEntity`.

Core flow:
1. Parse each subtask's ``.bddl`` file via ``bddl.parsing``.
2. Build an ``ArenaEntity`` scope and populate it with the entity positions from
   the task YAML.
3. Compile goal conditions using ``bddl.condition_evaluation.compile_state()``
   with the ``ArenaBackend``.
4. For every episode frame, update ``ArenaEntity.x/y`` and call
   ``bddl.condition_evaluation.evaluate_state()`` on the compiled conditions.
5. When a subtask's goal becomes True, record it as completed and advance.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

from bddl.condition_evaluation import compile_state, evaluate_state

from ..schema import EpisodeLog
from .arena_backend import ArenaBackend
from .arena_config import parse_arena_problem
from .arena_entity import ArenaEntity


@dataclass
class SubtaskResult:
    """Evaluation outcome for one subtask."""

    subtask_id: str
    description: str
    success: bool = False
    started_at_s: float | None = None
    completed_at_s: float | None = None
    duration_s: float | None = None
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "description": self.description,
            "success": self.success,
            "started_at_s": self.started_at_s,
            "completed_at_s": self.completed_at_s,
            "duration_s": self.duration_s,
            "timed_out": self.timed_out,
        }


@dataclass
class TaskResult:
    """Evaluation outcome for the entire task."""

    task_id: str
    description: str
    success: bool = False
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    subtasks_completed: int = 0
    subtasks_total: int = 0
    subtask_success_rate: float = 0.0
    total_duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "success": self.success,
            "subtasks_completed": self.subtasks_completed,
            "subtasks_total": self.subtasks_total,
            "subtask_success_rate": self.subtask_success_rate,
            "total_duration_s": self.total_duration_s,
            "subtask_results": [sr.to_dict() for sr in self.subtask_results],
        }


# ---------- Internal subtask definition ----------

@dataclass
class _SubtaskDef:
    """Parsed subtask: id, description, compiled BDDL goals, timeout."""
    id: str
    description: str
    compiled_goals: list  # list of HEAD nodes from compile_state
    timeout_s: float | None


# ---------- Evaluator ----------

class BDDLEvaluator:
    """Replay an EpisodeLog against a BDDL task definition (task.yaml + .bddl files).

    Usage::

        evaluator = BDDLEvaluator.from_task_dir("/path/to/multi_room_patrol")
        result = evaluator.evaluate(episode)
    """

    def __init__(
        self,
        task_id: str,
        description: str,
        entities: dict[str, ArenaEntity],
        subtask_defs: list[_SubtaskDef],
    ):
        self._task_id = task_id
        self._description = description
        self._entities = entities  # mutable — updated each tick
        self._subtask_defs = subtask_defs

    # ---- Factory: build from a task directory ----

    @classmethod
    def from_task_dir(cls, task_dir: str | pathlib.Path) -> "BDDLEvaluator":
        """Load a task.yaml + .bddl subtask files from a directory."""
        import yaml

        task_dir = pathlib.Path(task_dir)
        task_yaml = task_dir / "task.yaml"
        raw = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
        task_raw = raw.get("task", raw)

        task_id = task_raw.get("id", "unnamed")
        description = task_raw.get("description", "")

        # Build ArenaEntity objects from entity definitions
        entities: dict[str, ArenaEntity] = {}
        for eid, edata in task_raw.get("entities", {}).items():
            pos = edata.get("position")
            entities[eid] = ArenaEntity(
                name=eid,
                x=pos[0] if pos else 0.0,
                y=pos[1] if pos else 0.0,
            )

        backend = ArenaBackend()

        # Parse and compile each subtask
        subtask_defs: list[_SubtaskDef] = []
        for st_raw in task_raw.get("subtasks", []):
            bddl_file = task_dir / st_raw["bddl_file"]
            bddl_text = bddl_file.read_text(encoding="utf-8")

            _, parsed_objects, _, parsed_goals = parse_arena_problem(bddl_text)

            # Build scope: map BDDL object names → ArenaEntity instances
            scope: dict[str, ArenaEntity | None] = {}
            for cat, obj_names in parsed_objects.items():
                for oname in obj_names:
                    scope[oname] = entities.get(oname)

            compiled = compile_state(
                parsed_goals,
                backend,
                scope=scope,
                object_map=parsed_objects,
                generate_ground_options=False,
            )

            subtask_defs.append(_SubtaskDef(
                id=st_raw["id"],
                description=st_raw.get("description", ""),
                compiled_goals=compiled,
                timeout_s=st_raw.get("timeout_s"),
            ))

        return cls(
            task_id=task_id,
            description=description,
            entities=entities,
            subtask_defs=subtask_defs,
        )

    # ---- Core evaluation ----

    def evaluate(self, episode: EpisodeLog) -> TaskResult:
        """Evaluate a full episode against the task definition."""
        frames = episode.frames
        if not frames:
            return self._empty_result()

        subtask_results: list[SubtaskResult] = []
        current_idx = 0
        subtask_start_t: float | None = None
        t0 = frames[0].robot.t

        for frame in frames:
            t = frame.robot.t

            # Update the robot entity pose (in-place)
            robot = self._entities.get("robot_1")
            if robot is not None:
                robot.update_pose(frame.robot.x, frame.robot.y, frame.robot.yaw)

            if current_idx >= len(self._subtask_defs):
                continue

            stdef = self._subtask_defs[current_idx]
            if subtask_start_t is None:
                subtask_start_t = t

            # Timeout check
            if stdef.timeout_s is not None and (t - subtask_start_t) > stdef.timeout_s:
                subtask_results.append(SubtaskResult(
                    subtask_id=stdef.id,
                    description=stdef.description,
                    success=False,
                    started_at_s=subtask_start_t - t0,
                    completed_at_s=t - t0,
                    duration_s=t - subtask_start_t,
                    timed_out=True,
                ))
                current_idx += 1
                subtask_start_t = None
                continue

            # Evaluate using bddl 3.x
            success, _ = evaluate_state(stdef.compiled_goals)
            if success:
                subtask_results.append(SubtaskResult(
                    subtask_id=stdef.id,
                    description=stdef.description,
                    success=True,
                    started_at_s=subtask_start_t - t0,
                    completed_at_s=t - t0,
                    duration_s=t - subtask_start_t,
                ))
                current_idx += 1
                subtask_start_t = None

        # Handle remaining in-progress/not-started subtasks
        if current_idx < len(self._subtask_defs):
            stdef = self._subtask_defs[current_idx]
            t_last = frames[-1].robot.t
            subtask_results.append(SubtaskResult(
                subtask_id=stdef.id,
                description=stdef.description,
                success=False,
                started_at_s=(subtask_start_t - t0) if subtask_start_t is not None else None,
                completed_at_s=t_last - t0,
                duration_s=(t_last - subtask_start_t) if subtask_start_t is not None else None,
            ))
            for i in range(current_idx + 1, len(self._subtask_defs)):
                sd = self._subtask_defs[i]
                subtask_results.append(SubtaskResult(subtask_id=sd.id, description=sd.description))

        completed = sum(1 for sr in subtask_results if sr.success)
        total = len(self._subtask_defs)
        total_duration = frames[-1].robot.t - frames[0].robot.t

        return TaskResult(
            task_id=self._task_id,
            description=self._description,
            success=(completed == total),
            subtask_results=subtask_results,
            subtasks_completed=completed,
            subtasks_total=total,
            subtask_success_rate=completed / max(1, total),
            total_duration_s=total_duration,
        )

    def _empty_result(self) -> TaskResult:
        return TaskResult(
            task_id=self._task_id,
            description=self._description,
            subtasks_total=len(self._subtask_defs),
            subtask_results=[
                SubtaskResult(subtask_id=sd.id, description=sd.description)
                for sd in self._subtask_defs
            ],
        )
