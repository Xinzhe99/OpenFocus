"""Label model: a set of independently editable captions.

The old model held exactly two configurations (input stack, registered stack)
and "Add Label" overwrote whichever one matched the chosen target, so there
was no way to keep a source caption and add a second one, and no way to edit
or delete a single caption. A label now has a stable id, an explicit target
(input / registered / output) and its own enabled flag.

Persistence and rendering consume this model; the UI must not read the raw
config dictionaries.
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from utils.validators import LabelConfig, parse_label_range

TARGETS = ("input", "registered", "output")
_TARGET_ALIASES = {
    "input": "input",
    "registered": "registered",
    "output": "output",
    "result": "output",
    "fusion": "output",
}
_TARGET_STACK = {"input": 0, "registered": 1, "output": 1}


def normalize_target(value) -> str:
    """Map a stored/legacy target onto one of TARGETS."""
    text = str(value or "").strip().lower()
    return _TARGET_ALIASES.get(text, "registered")


@dataclass
class Label:
    """One caption with its own scope, style and frame range."""

    config: LabelConfig = field(default_factory=LabelConfig)
    label_id: str = ""

    def __post_init__(self) -> None:
        if not self.label_id:
            self.label_id = uuid.uuid4().hex
        if not self.config.label_id:
            self.config.label_id = self.label_id
        self.config.target = normalize_target(self.config.target)
        self.config.target_stack = _TARGET_STACK[self.config.target]

    @property
    def target(self) -> str:
        return self.config.target

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.config.enabled = bool(value)

    @property
    def range_valid(self) -> bool:
        return parse_label_range(self.config.range).valid

    @property
    def range_summary(self) -> str:
        parsed = parse_label_range(self.config.range)
        if not parsed.valid:
            return self.config.range or ""
        if parsed.all_frames:
            return "all"
        return self.config.range

    def copy(self) -> "Label":
        clone = Label(copy.deepcopy(self.config), self.label_id)
        return clone

    def to_dict(self) -> dict:
        data = dict(vars(self.config))
        data["label_id"] = self.label_id
        data["target"] = self.config.target
        data["enabled"] = self.enabled
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Label":
        cfg = LabelConfig()
        cfg.update_config(data)
        if "target" not in data:
            # Legacy projects only stored target_stack (0=input, 1=registered).
            legacy = int(data.get("target_stack", 1) or 0)
            cfg.target = "input" if legacy == 0 else "registered"
        label = cls(cfg, str(data.get("label_id") or ""))
        label.enabled = bool(data.get("enabled", True))
        return label


class LabelSet:
    """Ordered labels plus target/range queries used by every render path."""

    def __init__(self, labels: Optional[Iterable[Label]] = None) -> None:
        self._labels: List[Label] = list(labels or [])

    # ---------------------------------------------------------------- access
    def __iter__(self):
        return iter(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def all(self) -> List[Label]:
        return list(self._labels)

    def get(self, label_id: str) -> Optional[Label]:
        for label in self._labels:
            if label.label_id == label_id:
                return label
        return None

    def for_target(self, target: str) -> List[Label]:
        wanted = normalize_target(target)
        return [label for label in self._labels
                if label.enabled and label.target == wanted]

    def any_enabled(self, target: str) -> bool:
        return bool(self.for_target(target))

    # --------------------------------------------------------------- mutate
    def add(self, label: Label) -> Label:
        self._labels.append(label)
        return label

    def replace(self, label: Label) -> None:
        for index, existing in enumerate(self._labels):
            if existing.label_id == label.label_id:
                self._labels[index] = label
                return
        self._labels.append(label)

    def remove(self, label_id: str) -> bool:
        for index, existing in enumerate(self._labels):
            if existing.label_id == label_id:
                del self._labels[index]
                return True
        return False

    def set_enabled(self, target: str, enabled: bool) -> None:
        for label in self.for_target(target) if enabled else self._labels:
            if label.target == normalize_target(target):
                label.enabled = bool(enabled)

    def remove_target(self, target: str) -> int:
        wanted = normalize_target(target)
        before = len(self._labels)
        self._labels = [label for label in self._labels
                         if label.target != wanted]
        return before - len(self._labels)

    def clear(self) -> None:
        self._labels.clear()

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> List[dict]:
        return [label.to_dict() for label in self._labels]

    @classmethod
    def from_dict(cls, data) -> "LabelSet":
        labels = []
        for item in data or []:
            if isinstance(item, dict):
                labels.append(Label.from_dict(item))
        return cls(labels)

    def copy(self) -> "LabelSet":
        return LabelSet([label.copy() for label in self._labels])
