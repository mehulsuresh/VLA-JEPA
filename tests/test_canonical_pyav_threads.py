from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import starVLA.dataloader as dataloader_pkg
from starVLA.dataloader import canonical_subset_dataset as canonical


class _FakeCodecContext:
    def __init__(self):
        self.thread_count = 0


class _FakeStream:
    def __init__(self):
        self.average_rate = 30.0
        self.base_rate = None
        self.time_base = 1.0 / 30.0
        self.start_time = 0
        self.frames = 12
        self.duration = None
        self.thread_type = None
        self.codec_context = _FakeCodecContext()


class _FakeStreams:
    def __init__(self, stream):
        self.video = [stream]


class _FakeContainer:
    def __init__(self, stream):
        self.streams = _FakeStreams(stream)
        self.closed = False

    def close(self):
        self.closed = True


class _FakeAV:
    def __init__(self, container):
        self.container = container

    def open(self, path_key, mode="r"):
        assert path_key == "episode.mp4"
        assert mode == "r"
        return self.container


def _make_dataset(thread_count):
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.pyav_thread_count = thread_count
    dataset.pyav_thread_type = "SLICE"
    return dataset


def test_parse_pyav_thread_count_defaults_to_single_threaded():
    assert canonical._parse_pyav_thread_count(None) == 1
    assert canonical._parse_pyav_thread_count("1") == 1
    assert canonical._parse_pyav_thread_count("auto") == 1
    assert canonical._parse_pyav_thread_count("default") == 1
    assert canonical._parse_pyav_thread_count("0") == 0


def test_make_pyav_reader_caps_ffmpeg_decoder_threads(monkeypatch):
    stream = _FakeStream()
    container = _FakeContainer(stream)
    monkeypatch.setattr(canonical, "av", _FakeAV(container))

    reader = _make_dataset(thread_count=1)._make_pyav_reader("episode.mp4")

    assert reader.container is container
    assert reader.stream is stream
    assert stream.codec_context.thread_count == 1
    assert stream.thread_type == "SLICE"


def test_make_pyav_reader_can_leave_ffmpeg_auto_threads_enabled(monkeypatch):
    stream = _FakeStream()
    container = _FakeContainer(stream)
    monkeypatch.setattr(canonical, "av", _FakeAV(container))

    _make_dataset(thread_count=0)._make_pyav_reader("episode.mp4")

    assert stream.codec_context.thread_count == 0
    assert stream.thread_type == "SLICE"


def test_gcloud_file_copy_uses_atomic_temp_path(monkeypatch, tmp_path):
    destination = tmp_path / "episode.mp4"
    seen = {}

    def fake_run(cmd, **kwargs):
        temp_destination = Path(cmd[-1])
        seen["cmd"] = cmd
        assert temp_destination.parent == tmp_path
        assert temp_destination.name.startswith(".episode.mp4.")
        assert temp_destination.name.endswith(".tmp")
        temp_destination.write_text("downloaded", encoding="utf-8")

    monkeypatch.setattr(canonical.subprocess, "run", fake_run)

    canonical._run_gcloud_cp("gs://bucket/episode.mp4", destination)

    assert seen["cmd"][:3] == ["gcloud", "storage", "cp"]
    assert seen["cmd"][-2] == "gs://bucket/episode.mp4"
    assert destination.read_text(encoding="utf-8") == "downloaded"
    assert not list(tmp_path.glob("*.tmp"))


def test_gcloud_file_copy_cleans_failed_temp_path(monkeypatch, tmp_path):
    destination = tmp_path / "episode.mp4"

    def fake_run(cmd, **kwargs):
        temp_destination = Path(cmd[-1])
        temp_destination.write_text("partial", encoding="utf-8")
        raise canonical.subprocess.CalledProcessError(1, cmd, stderr="copy failed")

    monkeypatch.setattr(canonical.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Failed to copy canonical dataset shard"):
        canonical._run_gcloud_cp("gs://bucket/episode.mp4", destination)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_gcloud_file_copy_retries_timeout(monkeypatch, tmp_path):
    destination = tmp_path / "episode.mp4"
    attempts = []

    def fake_run(cmd, **kwargs):
        attempts.append(cmd)
        temp_destination = Path(cmd[-1])
        if len(attempts) == 1:
            temp_destination.write_text("partial", encoding="utf-8")
            raise canonical.subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        temp_destination.write_text("downloaded", encoding="utf-8")

    monkeypatch.setattr(canonical.subprocess, "run", fake_run)
    monkeypatch.setattr(canonical.time, "sleep", lambda _: None)

    canonical._run_gcloud_cp(
        "gs://bucket/episode.mp4",
        destination,
        timeout_seconds=1,
        retries=2,
        retry_backoff_seconds=0,
    )

    assert len(attempts) == 2
    assert destination.read_text(encoding="utf-8") == "downloaded"
    assert not list(tmp_path.glob("*.tmp"))


def test_canonical_worker_memory_budget_clamps_overcommit(monkeypatch):
    cfg = {
        "enforce_worker_memory_budget": True,
        "estimated_worker_memory_gb": 5.0,
        "worker_memory_budget_fraction": 0.65,
    }
    monkeypatch.setattr(dataloader_pkg, "_host_memory_gib", lambda: 62.0)
    monkeypatch.setattr(dataloader_pkg, "_distributed_world_size", lambda: 1)

    assert dataloader_pkg._maybe_clamp_canonical_workers_for_memory(cfg, 12) == 8

    cfg["enforce_worker_memory_budget"] = False
    assert dataloader_pkg._maybe_clamp_canonical_workers_for_memory(cfg, 12) == 12


def test_shard_data_prefetch_downloads_next_shard(monkeypatch, tmp_path):
    calls = []

    def fake_ensure_relative_path(**kwargs):
        calls.append(kwargs["relative_path"])
        return kwargs["root"] / kwargs["relative_path"]

    monkeypatch.setattr(canonical, "_ensure_relative_path", fake_ensure_relative_path)

    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.allow_gcs_download = True
    dataset.data_file_prefetch_shards = 1
    dataset.gcs_download_timeout_seconds = 900
    dataset.gcs_download_retries = 3
    dataset.gcs_download_retry_backoff_seconds = 0
    dataset._shard_prefetch_executor = None
    dataset._shard_prefetch_futures = canonical.OrderedDict()
    dataset._shard_prefetch_seen = set()
    dataset._decord_readers = canonical.OrderedDict()
    dataset._pyav_readers = canonical.OrderedDict()

    shard0_root = tmp_path / "shard0"
    shard1_root = tmp_path / "shard1"
    dataset.shards = [
        SimpleNamespace(
            root=shard0_root,
            gcs_prefix="gs://bucket/shard0",
            data_relative_path="data/chunk-000/file-000.parquet",
            data_path=shard0_root / "data/chunk-000/file-000.parquet",
            sidecar_path=shard0_root / "canonical_sidecars/data/chunk-000/file-000.npz",
        ),
        SimpleNamespace(
            root=shard1_root,
            gcs_prefix="gs://bucket/shard1",
            data_relative_path="data/chunk-000/file-001.parquet",
            data_path=shard1_root / "data/chunk-000/file-001.parquet",
            sidecar_path=shard1_root / "canonical_sidecars/data/chunk-000/file-001.npz",
        ),
    ]

    dataset._schedule_shard_data_prefetch(0)
    for future in list(dataset._shard_prefetch_futures.values()):
        assert future.result() is True

    assert calls == ["data/chunk-000/file-001.parquet"]
    dataset.close_video_readers()


def test_canonical_view_selection_uses_real_qwen_views_and_pads_vjepa():
    image_mapping = {
        "main": "observation.images.exterior_1_left",
        "right": "observation.images.wrist_left",
        "extra": "observation.images.exterior_2_left",
    }

    assert canonical._select_qwen_camera_slots(image_mapping) == ["main", "right", "extra"]
    assert canonical._select_vjepa_camera_slots(image_mapping) == ["right", "right", "main"]

    image_mapping = {"main": "observation.images.camera_top"}
    assert canonical._select_qwen_camera_slots(image_mapping) == ["main"]
    assert canonical._select_vjepa_camera_slots(image_mapping) == ["main", "main", "main"]


def test_canonical_sample_returns_qwen_frames_without_duplicate_qwen_views(tmp_path):
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.video_target_shift_steps = 1
    dataset._compact_offsets_cache = np.asarray([-1, 0, 1], dtype=np.int64)
    dataset.append_subtask_to_prompt = True
    dataset.subtask_prompt_separator = " | "

    main_path = tmp_path / "main.mp4"
    right_path = tmp_path / "right.mp4"
    extra_path = tmp_path / "extra.mp4"
    shard_data = SimpleNamespace(
        state=np.zeros((4, canonical.STATE_DIM), dtype=np.float32),
        action=np.zeros((4, canonical.ACTION_DIM), dtype=np.float32),
        action_mask=np.ones((4, canonical.ACTION_DIM), dtype=bool),
        episode_index=np.zeros(4, dtype=np.int64),
        frame_index=np.arange(4, dtype=np.int64),
        task_index=np.zeros(4, dtype=np.int64),
    )
    shard = canonical.ShardSpec(
        dataset_id="dataset",
        sid="sid",
        revision="main",
        adapter_group_id="adapter",
        adapter_path=tmp_path / "adapter.json",
        root=tmp_path,
        gcs_prefix="gs://bucket/sid/main",
        data_relative_path="data/chunk-000/file-000.parquet",
        data_path=tmp_path / "data.parquet",
        sidecar_path=tmp_path / "sidecar.npz",
        fps=30.0,
        camera_source_keys={
            "main": "observation.images.exterior_1_left",
            "right": "observation.images.wrist_left",
            "extra": "observation.images.exterior_2_left",
        },
        qwen_camera_slots=("main", "right", "extra"),
        vjepa_camera_slots=("right", "right", "main"),
        decode_camera_slots=("main", "right", "extra"),
        task_map={0: "task"},
        episodes=[],
    )
    episode = canonical.EpisodeSpec(
        local_start=0,
        length=4,
        task="task",
        video_paths={"main": main_path, "right": right_path, "extra": extra_path},
        video_base_frames={"main": 0, "right": 0, "extra": 0},
        subtask_spans=(
            canonical.SubtaskSpan("approach object", start_frame=0, end_frame=1),
            canonical.SubtaskSpan("grasp object", start_frame=1, end_frame=3),
        ),
    )
    context = {
        "shard_data": shard_data,
        "shard": shard,
        "episode": episode,
        "row_base": 1,
        "action_rows": np.asarray([1, 2], dtype=np.int64),
        "video_frames": {
            "main": (main_path, np.asarray([0, 1, 2], dtype=np.int64), tmp_path / "main.lock"),
            "right": (right_path, np.asarray([0, 1, 2], dtype=np.int64), tmp_path / "right.lock"),
            "extra": (extra_path, np.asarray([1], dtype=np.int64), tmp_path / "extra.lock"),
        },
        "qwen_frame_positions": {"main": 1, "right": 1, "extra": 0},
    }

    decoded_frames = {}
    for slot, path, base_value in (
        ("main", main_path, 10),
        ("right", right_path, 20),
        ("extra", extra_path, 30),
    ):
        frame_indices = (1,) if slot == "extra" else (0, 1, 2)
        decoded_frames[(slot, path.as_posix())] = {
            frame_index: np.full((2, 2, 3), base_value + frame_index, dtype=np.uint8)
            for frame_index in frame_indices
        }

    sample = dataset._sample_from_context(context, decoded_frames)

    assert sample["video_compact"].shape == (3, 3, 2, 2, 3)
    assert np.all(sample["video_compact"][0] == sample["video_compact"][1])
    assert np.all(sample["video_compact"][2, 1] == 11)
    assert sample["qwen_frames"].shape == (3, 2, 2, 3)
    assert np.all(sample["qwen_frames"][0] == 11)
    assert np.all(sample["qwen_frames"][1] == 21)
    assert np.all(sample["qwen_frames"][2] == 31)
    assert sample["qwen_view_slots"] == ("main", "right", "extra")
    assert sample["vjepa_view_slots"] == ("right", "right", "main")
    assert sample["qwen_vjepa_view_indices"].tolist() == [1, 1, 0]
    assert sample["lang"] == "task | grasp object"
    assert sample["subtask_label"] == "grasp object"


def test_canonical_getitems_retries_corrupt_video_batch(monkeypatch, tmp_path):
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.skip_corrupt_videos = True
    dataset.max_sample_decode_retries = 2
    dataset.total_windows = 1_000_000
    dataset._bad_video_paths = set()
    dataset._bad_video_warning_count = 0
    dataset.pyav_corrupt_warning_limit = 10

    bad_path = tmp_path / "bad.mp4"
    good_path = tmp_path / "good.mp4"
    shard = SimpleNamespace(root=tmp_path)
    attempts = []

    def fake_sample_context(index):
        path = bad_path if index in {1, 2} else good_path
        return {
            "sample_index": index,
            "shard": shard,
            "video_frames": {
                "main": (path, np.asarray([0, 1], dtype=np.int64), tmp_path / f"{path.stem}.lock")
            },
        }

    def fake_decode_video_frame_map(_shard, video_path, frame_indices, _lock_path):
        attempts.append(video_path.name)
        if video_path == bad_path:
            raise canonical._RecoverableVideoDecodeError("bad video", path_key=video_path.as_posix())
        return {int(index): np.zeros((2, 2, 3), dtype=np.uint8) for index in frame_indices.tolist()}

    monkeypatch.setattr(dataset, "_sample_context", fake_sample_context)
    monkeypatch.setattr(dataset, "_decode_video_frame_map", fake_decode_video_frame_map)
    monkeypatch.setattr(
        dataset,
        "_sample_from_context",
        lambda context, decoded_frames=None: {"sample_index": context["sample_index"]},
    )

    samples = dataset.__getitems__([1, 2])

    assert attempts == ["bad.mp4", "good.mp4"]
    assert bad_path.as_posix() in dataset._bad_video_paths
    assert [sample["sample_index"] for sample in samples] == [
        dataset._retry_index(1, 1),
        dataset._retry_index(2, 1),
    ]


def test_canonical_subtask_spans_parse_episode_metadata():
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.append_subtask_to_prompt = True

    spans = dataset._episode_subtask_spans(
        {
            "subtask_names": ["open drawer", "pick item"],
            "subtask_start_frames": [0, 15],
            "subtask_end_frames": [15, 42],
        }
    )

    assert spans == (
        canonical.SubtaskSpan("open drawer", start_frame=0, end_frame=15),
        canonical.SubtaskSpan("pick item", start_frame=15, end_frame=42),
    )
