from __future__ import annotations

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


def append_task_id_label_to_language(language: Any, task_id: Any, data_cfg: Any) -> tuple[str, str | None]:
    label = task_id_label_from_config(task_id, data_cfg)
    if label is None:
        return str(language).strip(), None
    return append_prompt_label(language, label, data_cfg), label
