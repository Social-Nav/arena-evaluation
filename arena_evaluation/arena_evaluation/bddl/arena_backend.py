"""Arena BDDL backend: bridges bddl 3.x predicate evaluation to ArenaEntity geometry.

Implements ``bddl.backend_abc.BDDLBackend`` so that ``bddl.condition_evaluation``
can compile and evaluate BDDL goal conditions using Arena's continuous-state
entities.
"""

from __future__ import annotations

from bddl.backend_abc import BDDLBackend
from bddl.logic_base import BinaryAtomicFormula, UnaryAtomicFormula


# ---------- Predicate classes ----------
# Each class follows the TrivialBackend pattern: inherit from
# BinaryAtomicFormula or UnaryAtomicFormula, set STATE_NAME, and implement
# _evaluate / _sample.


class ArenaNextToPredicate(BinaryAtomicFormula):
    STATE_NAME = "nextto"

    def _evaluate(self, obj1, obj2):
        return obj1.get_nextto(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass  # sampling not needed for evaluation-only


class ArenaInRoomPredicate(BinaryAtomicFormula):
    STATE_NAME = "inroom"

    def _evaluate(self, obj1, obj2):
        return obj1.get_inroom(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class ArenaTouchingPredicate(BinaryAtomicFormula):
    STATE_NAME = "touching"

    def _evaluate(self, obj1, obj2):
        return obj1.get_touching(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


# ---------- Backend ----------

_PREDICATE_MAP = {
    "nextto": ArenaNextToPredicate,
    "inroom": ArenaInRoomPredicate,
    "touching": ArenaTouchingPredicate,
}


class ArenaBackend(BDDLBackend):
    """BDDL backend that grounds predicates via ArenaEntity geometry."""

    def get_predicate_class(self, predicate_name: str):
        try:
            return _PREDICATE_MAP[predicate_name]
        except KeyError:
            raise KeyError(
                f"ArenaBackend does not support predicate '{predicate_name}'. "
                f"Available: {list(_PREDICATE_MAP.keys())}"
            )
