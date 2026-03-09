from __future__ import annotations

import json
import pathlib
import typing

from .schema import EpisodeLog


def load_episode(path: str | pathlib.Path) -> EpisodeLog:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    return EpisodeLog.from_dict(data)


def save_json(obj: typing.Any, path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")