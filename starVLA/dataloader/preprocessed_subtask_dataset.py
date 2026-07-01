from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from starVLA.dataloader.prompt_labels import append_task_id_label_to_language


ALL_CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


class PreprocessedSubtaskCollator:
    def __init__(
        self,
        model_id: str,
        prompt_template: str,
        replace_prompt: str,
        embodied_replace_prompt: str,
        special_action_token: str,
        max_action_tokens: int,
        embodied_action_token: str,
        state_replace_prompt: str = "",
        geometry_replace_prompt: str = "",
        extra_special_tokens: list[str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.replace_prompt = replace_prompt
        self.embodied_replace_prompt = embodied_replace_prompt
        self.special_action_token = special_action_token
        self.max_action_tokens = int(max_action_tokens)
        self.embodied_action_token = embodied_action_token
        self.state_replace_prompt = state_replace_prompt
        self.geometry_replace_prompt = geometry_replace_prompt
        self.extra_special_tokens = list(extra_special_tokens or [])
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            processor = AutoProcessor.from_pretrained(self.model_id)
            processor.tokenizer.padding_side = "left"
            action_tokens = [
                self.special_action_token.format(i) for i in range(self.max_action_tokens)
            ]
            special_tokens = [
                *action_tokens,
                self.embodied_action_token,
                *self.extra_special_tokens,
            ]
            new_tokens = [tok for tok in special_tokens if tok not in processor.tokenizer.get_vocab()]
            if new_tokens:
                processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
            self._processor = processor
        return self._processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        processor = self._get_processor()

        messages = []
        for sample in batch:
            prompt = self.prompt_template.replace("{instruction}", sample["lang"])
            prompt = prompt.replace("{actions}", self.replace_prompt)
            prompt = prompt.replace("{e_actions}", self.embodied_replace_prompt)
            prompt = prompt.replace("{state}", self.state_replace_prompt)
            prompt = prompt.replace("{geometry}", self.geometry_replace_prompt)
            content = [{"type": "image", "image": img} for img in sample["image"]]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        qwen_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        collated = {
            "qwen_inputs": qwen_inputs,
            "action": torch.from_numpy(
                np.asarray([sample["action"] for sample in batch], dtype=np.float32)
            ),
            "state": torch.from_numpy(
                np.asarray([sample["state"] for sample in batch], dtype=np.float32)
            ),
        }
        if "action_is_pad" in batch[0]:
            collated["action_is_pad"] = torch.from_numpy(
                np.asarray([sample["action_is_pad"] for sample in batch], dtype=bool)
            )

        if "video_compact" in batch[0]:
            compact_videos = np.stack([sample["video_compact"] for sample in batch]).transpose(0, 1, 2, 5, 3, 4)
            collated["video_compact"] = torch.from_numpy(np.ascontiguousarray(compact_videos))
        else:
            videos = np.stack([sample["video"] for sample in batch]).transpose(0, 1, 2, 5, 3, 4)
            collated["video"] = torch.from_numpy(np.ascontiguousarray(videos))
            if "video_target" in batch[0]:
                target_videos = np.stack([sample["video_target"] for sample in batch]).transpose(0, 1, 2, 5, 3, 4)
                collated["video_target"] = torch.from_numpy(np.ascontiguousarray(target_videos))

        tensor_keys = (
            "frame_index",
            "task_id",
            "future_task_id",
            "mistake_label",
            "future_mistake_label",
            "global_complexity_to_go",
            "future_global_complexity_to_go",
            "local_complexity_to_go",
            "future_local_complexity_to_go",
            "rabc_global_progress",
            "rabc_future_global_progress",
            "rabc_global_progress_delta",
            "rabc_stage_progress",
            "rabc_future_stage_progress",
            "rabc_progress_delta",
        )
        for key in tensor_keys:
            if not all(key in sample for sample in batch):
                continue
            values = [sample[key] for sample in batch]
            dtype = torch.long if key in {"frame_index", "task_id", "future_task_id"} else torch.float32
            collated[key] = torch.tensor(values, dtype=dtype)

        return collated


class PreprocessedSubtaskVLADataset(Dataset):
    def __init__(
        self,
        data_root_dir: str | Path,
        action_horizon: int,
        video_horizon: int,
        video_frame_stride: int = 1,
        video_target_shift_steps: int = 2,
        resolution_size: int = 224,
        video_resolution_size: int = 384,
        instruction_text: str = "Complete the task successfully.",
        current_cameras: list[str] | None = None,
        frame_cache_size: int = 256,
        data_cfg: Any | None = None,
    ) -> None:
        self.root = Path(data_root_dir)
        self.action_horizon = int(action_horizon)
        self.video_horizon = int(video_horizon)
        self.video_frame_stride = max(int(video_frame_stride), 1)
        self.video_target_shift_steps = max(int(video_target_shift_steps), 1)
        if self.video_horizon <= self.video_target_shift_steps:
            raise ValueError(
                "video_horizon must be greater than video_target_shift_steps so the context clip is non-empty"
            )
        self.video_context_horizon = self.video_horizon - self.video_target_shift_steps
        self.resolution_size = int(resolution_size)
        self.video_resolution_size = int(video_resolution_size)
        self.instruction_text = instruction_text
        self.data_cfg = data_cfg
        self.current_cameras = list(current_cameras or ALL_CAMERAS)
        self.frame_cache_size = max(int(frame_cache_size), 0)

        self.episodes: dict[str, dict[str, Any]] = {}
        self.records: list[tuple[str, int]] = []
        self._bad_image_warned: set[str] = set()
        self._frame_cache: OrderedDict[tuple[str, str, int], np.ndarray] = OrderedDict()

        episode_dirs = sorted(
            [p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("episode_")]
        )
        if not episode_dirs:
            raise RuntimeError(f"No episode_* directories found under {self.root}")

        for episode_dir in episode_dirs:
            metadata_path = episode_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            labels = metadata.get("labels", {})
            features = metadata.get("frame_features", {})
            frame_indices = metadata.get("frame_indices", [])
            if not frame_indices:
                continue
            if "action" not in features or "observation.state" not in features:
                continue

            frame_indices = np.asarray(frame_indices, dtype=np.int32)
            actions = np.asarray(features["action"], dtype=np.float32)
            states = np.asarray(features["observation.state"], dtype=np.float32)
            base_length = min(len(frame_indices), len(actions), len(states))
            if base_length <= 0:
                continue

            task_ids, has_task_labels = self._optional_label_array(
                labels,
                "subtask_id",
                dtype=np.int32,
                length=base_length,
                default_value=0,
            )
            raw_mistake_labels, has_mistake_labels = self._optional_label_array(
                labels,
                "mistake_label",
                dtype=np.float32,
                length=base_length,
                default_value=0.0,
            )
            if has_mistake_labels:
                mistake_labels = 1.0 - np.clip(raw_mistake_labels, 0.0, 1.0)
            else:
                mistake_labels = np.zeros(base_length, dtype=np.float32)
            global_ctg, has_global_ctg = self._optional_label_array(
                labels,
                "global_complexity_to_go",
                dtype=np.float32,
                length=base_length,
                default_value=np.nan,
            )
            local_ctg, has_local_ctg = self._optional_label_array(
                labels,
                "local_complexity_to_go",
                dtype=np.float32,
                length=base_length,
                default_value=np.nan,
            )

            present_label_lengths = [
                len(array)
                for array, present in (
                    (task_ids, has_task_labels),
                    (raw_mistake_labels, has_mistake_labels),
                    (global_ctg, has_global_ctg),
                    (local_ctg, has_local_ctg),
                )
                if present
            ]
            length = min([base_length, *present_label_lengths])
            if length <= 0:
                continue

            self.episodes[episode_dir.name] = {
                "dir": episode_dir,
                "frame_indices": frame_indices[:length],
                "action": actions[:length],
                "state": states[:length],
                "task_id": task_ids[:length],
                "mistake_label": mistake_labels[:length],
                "global_complexity_to_go": global_ctg[:length],
                "local_complexity_to_go": local_ctg[:length],
                "has_task_labels": has_task_labels,
                "has_mistake_labels": has_mistake_labels,
                "has_global_complexity_to_go": has_global_ctg,
                "has_local_complexity_to_go": has_local_ctg,
            }
            self.records.extend((episode_dir.name, i) for i in range(length))

        if not self.records:
            raise RuntimeError(f"No usable records found under {self.root}")

        first_episode = next(iter(self.episodes.values()))
        self.action_dim = int(first_episode["action"].shape[-1])
        self.state_dim = int(first_episode["state"].shape[-1])

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _optional_label_array(
        labels: dict[str, Any],
        key: str,
        *,
        dtype: Any,
        length: int,
        default_value: float | int,
    ) -> tuple[np.ndarray, bool]:
        if key in labels:
            return np.asarray(labels[key], dtype=dtype), True
        return np.full(length, default_value, dtype=dtype), False

    def _resolve_sampled_frame_index(self, row_idx: int, offset: int, frame_indices: np.ndarray) -> int:
        sampled_row_idx = min(max(row_idx + offset * self.video_frame_stride, 0), len(frame_indices) - 1)
        return int(frame_indices[sampled_row_idx])

    def _load_rgb(self, episode_name: str, camera: str, frame_index: int) -> np.ndarray:
        cache_key = (episode_name, camera, int(frame_index))
        if self.frame_cache_size > 0:
            cached = self._frame_cache.get(cache_key)
            if cached is not None:
                self._frame_cache.move_to_end(cache_key)
                return cached.copy()

        episode_dir = self.episodes[episode_name]["dir"]
        camera_key = camera.replace(".", "_")
        frame_path = episode_dir / f"{camera_key}_f{frame_index:06d}.jpg"
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            frame_key = str(frame_path)
            if frame_key not in self._bad_image_warned:
                self._bad_image_warned.add(frame_key)
            return np.zeros((self.video_resolution_size, self.video_resolution_size, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.frame_cache_size > 0:
            self._frame_cache[cache_key] = rgb
            self._frame_cache.move_to_end(cache_key)
            while len(self._frame_cache) > self.frame_cache_size:
                self._frame_cache.popitem(last=False)
        return rgb

    def _resize_rgb(self, image: np.ndarray, size: int) -> np.ndarray:
        h, w = image.shape[:2]
        if h == size and w == size:
            return image
        # INTER_AREA is faster and higher quality for downsampling
        interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
        return cv2.resize(image, (size, size), interpolation=interp)

    @staticmethod
    def _safe_progress(value: float, default: float = 0.0) -> float:
        if value is None:
            return default
        value = float(value)
        if np.isnan(value):
            return default
        return float(np.clip(value, 0.0, 1.0))

    def save_dataset_statistics(self, output_path: str | Path) -> None:
        task_ids = []
        mistake_sum = 0.0
        episodes_with_task_labels = 0
        episodes_with_mistake_labels = 0
        episodes_with_global_ctg = 0
        episodes_with_local_ctg = 0
        for episode in self.episodes.values():
            if episode["has_task_labels"]:
                task_ids.extend(np.unique(episode["task_id"]).tolist())
                episodes_with_task_labels += 1
            if episode["has_mistake_labels"]:
                mistake_sum += float(np.sum(episode["mistake_label"]))
                episodes_with_mistake_labels += 1
            if episode["has_global_complexity_to_go"]:
                episodes_with_global_ctg += 1
            if episode["has_local_complexity_to_go"]:
                episodes_with_local_ctg += 1
        stats = {
            "num_records": len(self.records),
            "num_episodes": len(self.episodes),
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "current_cameras": list(self.current_cameras),
            "unique_task_ids": sorted(set(int(x) for x in task_ids)),
            "mistake_positive_count": int(mistake_sum),
            "episodes_with_task_labels": episodes_with_task_labels,
            "episodes_with_mistake_labels": episodes_with_mistake_labels,
            "episodes_with_global_complexity_to_go": episodes_with_global_ctg,
            "episodes_with_local_complexity_to_go": episodes_with_local_ctg,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_name, row_idx = self.records[index]
        episode = self.episodes[episode_name]
        frame_indices = episode["frame_indices"]

        current_frame_idx = int(frame_indices[row_idx])
        image_list = []
        union_offsets = list(range(-(self.video_context_horizon - 1), self.video_target_shift_steps + 1))
        compact_video_views = []
        for camera in self.current_cameras:
            current_rgb = self._load_rgb(episode_name, camera, current_frame_idx)
            image_list.append(
                Image.fromarray(self._resize_rgb(current_rgb, self.resolution_size))
            )

            merged_frames = []
            for offset in union_offsets:
                sampled_frame_idx = self._resolve_sampled_frame_index(row_idx, offset, frame_indices)
                rgb = current_rgb if sampled_frame_idx == current_frame_idx else self._load_rgb(
                    episode_name, camera, sampled_frame_idx
                )
                merged_frames.append(self._resize_rgb(rgb, self.video_resolution_size))
            merged_frames = np.stack(merged_frames, axis=0)
            compact_video_views.append(merged_frames)

        if len(compact_video_views) == 1:
            image_list = [image_list[0], image_list[0].copy()]
            compact_video_views = [compact_video_views[0], compact_video_views[0].copy()]

        action_rows = []
        action_is_pad = []
        for offset in range(self.action_horizon):
            requested_idx = row_idx + offset
            future_idx = min(requested_idx, len(frame_indices) - 1)
            action_rows.append(episode["action"][future_idx])
            action_is_pad.append(requested_idx >= len(frame_indices))
        action = np.stack(action_rows, axis=0).astype(np.float32, copy=False)
        state = episode["state"][row_idx : row_idx + 1].astype(np.float32, copy=False)

        future_idx = min(row_idx + self.action_horizon - 1, len(frame_indices) - 1)
        current_task = int(episode["task_id"][row_idx])
        future_task = int(episode["task_id"][future_idx])
        current_local_ctg = float(episode["local_complexity_to_go"][row_idx])
        future_local_ctg = float(episode["local_complexity_to_go"][future_idx])
        current_global_ctg = float(episode["global_complexity_to_go"][row_idx])
        future_global_ctg = float(episode["global_complexity_to_go"][future_idx])
        current_mistake = float(episode["mistake_label"][row_idx])
        future_mistake = float(episode["mistake_label"][future_idx])

        stage_min = int(np.min(episode["task_id"]))
        stage_max = int(np.max(episode["task_id"]))
        num_stages = max(stage_max - stage_min + 1, 1)
        current_stage_idx = current_task - stage_min
        future_stage_idx = future_task - stage_min
        current_local_progress = 1.0 - self._safe_progress(current_local_ctg, 1.0)
        future_local_progress = 1.0 - self._safe_progress(future_local_ctg, 1.0)
        current_stage_progress = (current_stage_idx + current_local_progress) / num_stages
        future_stage_progress = (future_stage_idx + future_local_progress) / num_stages
        current_global_progress = 1.0 - self._safe_progress(current_global_ctg, 1.0)
        future_global_progress = 1.0 - self._safe_progress(future_global_ctg, 1.0)
        language, task_id_label = append_task_id_label_to_language(
            self.instruction_text,
            current_task,
            self.data_cfg,
        )

        sample = {
            "action": action,
            "image": image_list,
            "lang": language,
            "video_compact": np.stack(compact_video_views, axis=0),
            "state": state,
            "action_is_pad": np.asarray(action_is_pad, dtype=bool),
            "frame_index": current_frame_idx,
            "task_id": current_task,
            "future_task_id": future_task,
            "mistake_label": current_mistake,
            "future_mistake_label": future_mistake,
            "global_complexity_to_go": current_global_ctg,
            "future_global_complexity_to_go": future_global_ctg,
            "local_complexity_to_go": current_local_ctg,
            "future_local_complexity_to_go": future_local_ctg,
        }
        if task_id_label is not None:
            sample["task_id_label"] = task_id_label
        if episode["has_global_complexity_to_go"]:
            sample["rabc_global_progress"] = current_global_progress
            sample["rabc_future_global_progress"] = future_global_progress
            sample["rabc_global_progress_delta"] = future_global_progress - current_global_progress
        else:
            sample["rabc_global_progress"] = np.nan
            sample["rabc_future_global_progress"] = np.nan
            sample["rabc_global_progress_delta"] = np.nan
        if episode["has_task_labels"] and episode["has_local_complexity_to_go"]:
            sample["rabc_stage_progress"] = current_stage_progress
            sample["rabc_future_stage_progress"] = future_stage_progress
            sample["rabc_progress_delta"] = future_stage_progress - current_stage_progress
        else:
            sample["rabc_stage_progress"] = np.nan
            sample["rabc_future_stage_progress"] = np.nan
            sample["rabc_progress_delta"] = np.nan
        return sample
