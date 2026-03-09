# BDDL Evaluation Framework — Usage Guide

## Overview

This module provides BDDL (Behavior Domain Definition Language) evaluation for long-horizon VLN tasks in Arena-Rosnav. It uses the official `bddl` 3.x package from Stanford BEHAVIOR project, bridged to Arena's simulation state via a custom `ArenaBackend`.

**Architecture:**
```
Isaac Sim USD scene ──> /isaac/GetPrims service ──> ArenaEntity.update_pose()
/odom topic ────────────────────────────────────> ArenaEntity.update_pose() (robot)
                                                        │
                                              ArenaBackend predicates
                                              (nextto, touching, inroom)
                                                        │
                                              bddl.condition_evaluation
                                              evaluate_state() → bool
```

## Quick Start

### 1. Standalone Test (no simulator needed)

Verifies the full BDDL pipeline: parsing → compile → evaluate.

```bash
cd ~/arena5_ws
source arena

cd src/Arena/arena_evaluation/arena_evaluation
python -m arena_evaluation.bddl.test_world_state --standalone
```

**Expected output:**
```
BDDL World State Test — STANDALONE MODE (no ROS 2)
Task: multi_room_patrol
Subtasks: 3
Entities: ['robot_1', 'kitchen_table_1', 'living_room_sofa_1', 'bedroom_desk_1']

>>> Moving robot to kitchen_table_1 (5.0, 3.0) ...
  (robot_1, kitchen_table_1)   nextto=True   touching=True   dist=0.000
  subtask[0] goto_kitchen: SATISFIED

STANDALONE TEST PASSED — BDDL pipeline is functional
```

### 2. Live Test with Arena

Launch Arena with a simulator, then start the BDDL evaluator node. The evaluator subscribes to odom for robot pose and optionally calls `/isaac/GetPrims` for ground-truth object poses.

**Terminal 1 — Launch Arena:**
```bash
cd ~/arena5_ws && source arena
DISPLAY=:10 arena launch sim:=isaac headless:=1 world:=map_empty
```

**Terminal 2 — Start the evaluator node:**
```bash
cd ~/arena5_ws && source arena
ros2 run arena_evaluation arena-bddl-evaluator \
  --ros-args \
  -p task_dir:=$(ros2 pkg prefix arena_evaluation)/lib/python3.10/dist-packages/arena_evaluation/bddl/tasks/multi_room_patrol \
  -p odom_topic:=odom \
  -p eval_rate_hz:=10.0 \
  -p use_isaac_gt:=true \
  -p verbose:=true
```

The odom topic is auto-discovered (scans for `*/odom` with type `nav_msgs/msg/Odometry`). You can also set it explicitly, e.g. `-p odom_topic:=/task_generator_node/jackal/odom`.

When `verbose:=true`, the node prints the full PDDL world state to the terminal every 3 seconds:

```
============================================================
BDDL World State (verbose)
============================================================

  Entity                         X        Y      Yaw   Radius
  ------------------------- -------- -------- -------- --------
  robot_1                      3.500    9.500    2.314     0.30
  kitchen_table_1              5.000    3.000    0.000     0.50
  living_room_sofa_1          10.000    7.000    0.000     0.50
  bedroom_desk_1              15.000    2.000    0.000     0.50

  Pair                                          nextto   touching     dist
  --------------------------------------------- -------- ---------- --------
  (robot_1, kitchen_table_1)                     False      False    6.671
  (robot_1, living_room_sofa_1)                  False      False    6.800
  ...

--- Subtask Goal Evaluation ---
  [0] goto_kitchen: NOT SATISFIED
  [1] goto_living_room: NOT SATISFIED
  [2] goto_bedroom: NOT SATISFIED
```

**Monitor status (from any terminal):**
```bash
ros2 topic echo /bddl_evaluator/bddl_status
```

**Get final result:**
```bash
ros2 service call /bddl_evaluator/bddl_result std_srvs/srv/Trigger
```

### 3. Offline Episode Evaluation (CLI)

Evaluate a recorded episode JSON against a BDDL task:

```bash
arena-eval bddl-eval /path/to/task/directory /path/to/episode.json
```

## How to Check Results

### World State Output

Each verbose print (or status topic message) contains:

| Column     | Meaning                                    |
|------------|--------------------------------------------|
| `Entity`   | Object name (robot_1, kitchen_table_1, ..) |
| `X`, `Y`   | Current position in world frame            |
| `Yaw`      | Heading in radians                         |
| `nextto`   | `True` if distance ≤ radius_a + radius_b + 0.5m |
| `touching` | `True` if distance ≤ radius_a + radius_b  |
| `dist`     | Euclidean distance between entity pair     |

### Subtask Goal Evaluation

```
[0] goto_kitchen:     SATISFIED       ← robot reached kitchen_table_1
[1] goto_living_room: NOT SATISFIED   ← robot not yet near sofa
[2] goto_bedroom:     NOT SATISFIED   ← robot not yet near desk
```

A subtask is `SATISFIED` when all its BDDL goal conditions evaluate to `True`.

### Final Task Result (JSON)

```json
{
  "task_id": "multi_room_patrol",
  "success": true,
  "subtasks_completed": 3,
  "subtasks_total": 3,
  "subtask_success_rate": 1.0,
  "total_duration_s": 12.7,
  "subtask_results": [
    {"subtask_id": "goto_kitchen", "success": true, "duration_s": 3.1},
    {"subtask_id": "goto_living_room", "success": true, "duration_s": 4.1},
    {"subtask_id": "goto_bedroom", "success": true, "duration_s": 4.6}
  ]
}
```

Key fields to check:
- **`success`**: `true` only if ALL subtasks completed
- **`subtask_success_rate`**: 0.0–1.0, fraction of subtasks completed
- **`timed_out`**: appears in subtask results if it exceeded `timeout_s`

## Creating New Tasks

### Directory Structure

```
tasks/my_task/
├── task.yaml          # Orchestration: entities, subtask sequence
├── subtask0.bddl      # BDDL goal for subtask 0
├── subtask1.bddl      # BDDL goal for subtask 1
└── ...
```

### task.yaml

```yaml
task:
  id: my_task
  description: "Navigate to A then B"

  entities:
    robot_1:
      type: robot
    waypoint_a:
      type: object
      position: [3.0, 4.0]
    waypoint_b:
      type: object
      position: [8.0, 1.0]

  subtasks:
    - id: goto_a
      description: "Go to waypoint A"
      bddl_file: subtask0.bddl
      timeout_s: 30.0

    - id: goto_b
      description: "Go to waypoint B"
      bddl_file: subtask1.bddl
      timeout_s: 30.0
```

### .bddl File

```lisp
(define (problem my_task_subtask0)
    (:domain arena)
    (:objects robot_1 waypoint_a - object)
    (:init)
    (:goal (and (nextto ?robot_1 ?waypoint_a)))
)
```

### Available Predicates

| Predicate | Type   | Condition                          |
|-----------|--------|------------------------------------|
| `nextto`  | Binary | dist ≤ radius_a + radius_b + 0.5m |
| `touching`| Binary | dist ≤ radius_a + radius_b        |
| `inroom`  | Binary | dist ≤ room_radius (obj2 must be room) |

## File Reference

```
arena_evaluation/bddl/
├── arena_backend.py       # BDDLBackend implementation (predicate → ArenaEntity)
├── arena_config.py        # Monkey-patches bddl domain resolver for "arena" domain
├── arena_entity.py        # Mutable entity wrapper with pose + predicate methods
├── evaluator.py           # Core BDDLEvaluator: offline episode replay
├── demo.py                # Synthetic trajectory demo
├── test_world_state.py    # Integration test (standalone / ROS 2 / Isaac Sim)
├── definitions/
│   └── domain_arena.bddl  # Arena PDDL domain (predicates: nextto, inroom, touching)
└── tasks/
    └── multi_room_patrol/  # Example task
        ├── task.yaml
        ├── subtask0.bddl
        ├── subtask1.bddl
        └── subtask2.bddl

arena_evaluation/ros/
└── bddl_evaluator_node.py  # Online ROS 2 evaluator node (with --verbose support)
```
