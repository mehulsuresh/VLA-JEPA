from __future__ import annotations

import random
from typing import Any


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            value = getter(key)
            return default if value is None else value
    return getattr(config, key, default)


def _label_map_key(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def normalize_label_map(label_map: Any) -> dict[str, str]:
    if label_map is None:
        return {}
    items = label_map.items() if hasattr(label_map, "items") else []
    normalized = {}
    for key, value in items:
        text = str(value).strip()
        if text:
            normalized[_label_map_key(key)] = text
    return normalized


def task_id_label_from_config(task_id: Any, data_cfg: Any) -> str | None:
    if not bool(cfg_get(data_cfg, "append_task_id_to_prompt", False)):
        return None

    label_map = normalize_label_map(cfg_get(data_cfg, "task_id_label_map", {}))
    label = label_map.get(_label_map_key(task_id))
    if label is None:
        fallback_template = cfg_get(data_cfg, "task_id_prompt_fallback_template", None)
        if not fallback_template:
            return None
        label = str(fallback_template).format(task_id=task_id)

    label = str(label).strip()
    if not label:
        return None

    template = str(cfg_get(data_cfg, "task_id_prompt_template", "{label}"))
    return template.format(task_id=task_id, label=label).strip()


def _task_id_prompt_append_probability(data_cfg: Any) -> float:
    value = cfg_get(
        data_cfg,
        "task_id_prompt_append_probability",
        cfg_get(data_cfg, "subtask_prompt_append_probability", 1.0),
    )
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = 1.0
    if probability > 1.0:
        probability = probability / 100.0
    return max(0.0, min(1.0, probability))


def _ignored_prompt_labels(data_cfg: Any) -> set[str]:
    values = cfg_get(data_cfg, "subtask_prompt_ignored_labels", ("__unlabeled__",))
    if values is None:
        return set()
    if isinstance(values, str):
        values = (values,)
    try:
        return {str(value).strip().casefold() for value in values if str(value).strip()}
    except TypeError:
        text = str(values).strip()
        return {text.casefold()} if text else set()


def append_prompt_label(language: Any, label: str, data_cfg: Any) -> str:
    task = str(language).strip()
    label = str(label).strip()
    if not label:
        return task

    separator = str(
        cfg_get(
            data_cfg,
            "task_id_prompt_separator",
            cfg_get(data_cfg, "subtask_prompt_separator", " | "),
        )
    )
    if separator in task and task.rsplit(separator, 1)[-1].strip() == label:
        return task
    return f"{task}{separator}{label}"


def append_resolved_label_to_language(language: Any, label: Any, data_cfg: Any) -> tuple[str, str | None]:
    if not bool(cfg_get(data_cfg, "append_task_id_to_prompt", False)):
        return str(language).strip(), None

    if label is None:
        return str(language).strip(), None
    label = str(label).strip()
    if not label:
        return str(language).strip(), None
    if label.casefold() in _ignored_prompt_labels(data_cfg):
        return str(language).strip(), None

    if random.random() >= _task_id_prompt_append_probability(data_cfg):
        return str(language).strip(), None
    return append_prompt_label(language, label, data_cfg), label


def append_task_id_label_to_language(language: Any, task_id: Any, data_cfg: Any) -> tuple[str, str | None]:
    label = task_id_label_from_config(task_id, data_cfg)
    return append_resolved_label_to_language(language, label, data_cfg)
