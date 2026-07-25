from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import starVLA.dataloader as dataloader_pkg
from starVLA.dataloader import canonical_subset_dataset as canonical


CANONICAL_CONFIG_PATHS = sorted(
    Path(__file__).resolve().parents[1]
    .joinpath("scripts/config")
    .glob("vlajepa_robot_ft_canonical*.yaml")
)


class _FakeCodecContext:
    def __init__(self):
        self.thread_count = 0


def test_all_canonical_configs_use_explicit_subtask_prompt_contract():
    assert len(CANONICAL_CONFIG_PATHS) == 6
    expected = {
        "subtask_prompt_source_column": "subtask_index",
        "subtask_prompt_label_column": "local_subtask_text",
        "subtask_prompt_separator": " | ",
        "subtask_prompt_ignored_labels": ["__unlabeled__"],
    }

    for config_path in CANONICAL_CONFIG_PATHS:
        data_cfg = OmegaConf.to_container(
            OmegaConf.load(config_path).datasets.vla_data,
            resolve=True,
        )
        assert {key: data_cfg[key] for key in expected} == expected, config_path
        expected_enabled = "canonical_full_a100x8" in config_path.name
        assert data_cfg["append_subtask_to_prompt"] is expected_enabled, config_path
        assert data_cfg["subtask_prompt_append_probability"] == 0.7, config_path


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


class _FakeDecodeError(Exception):
    pass


class _FakeDecodeFrame:
    def __init__(self, index):
        self.index = index
        self.pts = index
        self.time = index / 30.0

    def to_ndarray(self, format="rgb24"):
        assert format == "rgb24"
        return np.full((2, 2, 3), self.index, dtype=np.uint8)


class _FakeDecodeContainer:
    def __init__(self):
        self.seek_pts = None
        self.closed = False

    def seek(self, pts, stream=None, backward=True, any_frame=False):
        self.seek_pts = int(pts)

    def decode(self, stream):
        if self.seek_pts == 0:
            for index in range(11):
                yield _FakeDecodeFrame(index)
        raise _FakeDecodeError("decode failed")

    def close(self):
        self.closed = True


def _make_dataset(thread_count):
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.pyav_thread_count = thread_count
    dataset.pyav_thread_type = "SLICE"
    return dataset


def _make_pyav_decode_dataset(max_fill_distance):
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.pyav_reader_cache_size = 0
    dataset.pyav_decode_retry_extra_frames = 120
    dataset.pyav_max_nearest_fill_distance = max_fill_distance
    dataset.skip_corrupt_videos = False
    dataset.pyav_max_missing_frames_for_fill = 0
    dataset.pyav_fail_on_decode_error_recovery = True
    dataset.pyav_corrupt_warning_limit = 0
    dataset._pyav_corrupt_warning_count = 0
    dataset._bad_video_paths = set()
    dataset.video_resolution_size = 2

    def make_reader(path_key):
        container = _FakeDecodeContainer()
        return SimpleNamespace(
            container=container,
            stream=object(),
            fps=30.0,
            time_base=1.0 / 30.0,
            start_time=0,
            frame_count=1000,
        )

    dataset._get_pyav_reader = make_reader
    dataset._make_pyav_reader = make_reader
    dataset._drop_pyav_reader = lambda path_key: None
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


def test_pyav_nearest_frame_fill_rejects_distant_candidates(monkeypatch):
    monkeypatch.setattr(canonical, "av", object())
    dataset = _make_pyav_decode_dataset(max_fill_distance=2)

    with pytest.raises(RuntimeError, match="nearest-frame fill exceeded safety distance"):
        dataset._decode_video_pyav("episode.mp4", np.asarray([100], dtype=np.int64))


def test_pyav_nearest_frame_fill_allows_small_distance(monkeypatch):
    monkeypatch.setattr(canonical, "av", object())
    dataset = _make_pyav_decode_dataset(max_fill_distance=2)

    frames = dataset._decode_video_pyav("episode.mp4", np.asarray([12], dtype=np.int64))

    assert frames.shape == (1, 2, 2, 3)
    assert np.all(frames[0] == 10)


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


def test_canonical_worker_memory_budget_rejects_overcommit(monkeypatch):
    cfg = {
        "enforce_worker_memory_budget": True,
        "estimated_worker_memory_gb": 5.0,
        "worker_memory_budget_fraction": 0.65,
    }
    monkeypatch.setattr(dataloader_pkg, "_host_memory_gib", lambda: 62.0)
    monkeypatch.setattr(dataloader_pkg, "_distributed_world_size", lambda: 1)

    with pytest.raises(ValueError, match="num_workers exceeds"):
        dataloader_pkg._maybe_clamp_canonical_workers_for_memory(cfg, 12)

    cfg["enforce_worker_memory_budget"] = False
    assert dataloader_pkg._maybe_clamp_canonical_workers_for_memory(cfg, 12) == 12


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("prefetch_factor", 0),
        ("prefetch_factor", -1),
        ("prefetch_factor", True),
        ("prefetch_factor", "2"),
        ("worker_torch_threads", 0),
        ("worker_cv2_threads", -1),
    ),
)
def test_worker_knobs_fail_instead_of_being_clamped(key, value):
    with pytest.raises(ValueError, match=key):
        dataloader_pkg._positive_loader_integer({key: value}, key)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("estimated_worker_memory_gb", 0.0),
        ("estimated_worker_memory_gb", float("inf")),
        ("worker_memory_budget_fraction", 0.0),
        ("worker_memory_budget_fraction", 1.01),
    ),
)
def test_canonical_worker_memory_values_fail_instead_of_being_clamped(
    monkeypatch,
    field,
    value,
):
    cfg = {
        "enforce_worker_memory_budget": True,
        "estimated_worker_memory_gb": 5.0,
        "worker_memory_budget_fraction": 0.65,
    }
    cfg[field] = value
    monkeypatch.setattr(dataloader_pkg, "_host_memory_gib", lambda: 62.0)

    with pytest.raises(ValueError, match=field):
        dataloader_pkg._maybe_clamp_canonical_workers_for_memory(cfg, 1)


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


def test_canonical_subtask_prompt_defaults_disabled_and_accepts_logical_contract():
    assert canonical._canonical_subtask_prompt_settings({}) == (
        False,
        "subtask_index",
        "local_subtask_text",
    )
    assert canonical._canonical_subtask_prompt_settings(
        {
            "append_subtask_to_prompt": True,
            "subtask_prompt_source_column": "subtask_index",
            "subtask_prompt_label_column": "local_subtask_text",
        }
    ) == (True, "subtask_index", "local_subtask_text")


@pytest.mark.parametrize(
    ("config_key", "invalid_value", "expected_message"),
    (
        (
            "subtask_prompt_source_column",
            "local_stage_id",
            "subtask_prompt_source_column='subtask_index'",
        ),
        (
            "subtask_prompt_label_column",
            "description",
            "subtask_prompt_label_column='local_subtask_text'",
        ),
    ),
)
def test_canonical_subtask_prompt_rejects_noncanonical_logical_columns(
    config_key,
    invalid_value,
    expected_message,
):
    config = {
        "append_subtask_to_prompt": True,
        "subtask_prompt_source_column": "subtask_index",
        "subtask_prompt_label_column": "local_subtask_text",
        config_key: invalid_value,
    }

    with pytest.raises(ValueError, match=expected_message):
        canonical._canonical_subtask_prompt_settings(config)


def test_canonical_subtask_prompt_requires_usable_selected_span():
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.append_subtask_to_prompt = True
    dataset.data_cfg = {"subtask_prompt_ignored_labels": ["__unlabeled__"]}
    ignored_episode = SimpleNamespace(
        subtask_spans=(canonical.SubtaskSpan("__unlabeled__"),)
    )
    dataset.shards = [SimpleNamespace(episodes=[ignored_episode])]

    with pytest.raises(RuntimeError, match="zero usable nonignored subtask spans"):
        dataset._validate_subtask_prompt_coverage()

    labeled_episode = SimpleNamespace(
        subtask_spans=(canonical.SubtaskSpan("pick up the chain"),)
    )
    dataset.shards.append(SimpleNamespace(episodes=[labeled_episode]))
    dataset._validate_subtask_prompt_coverage()


def test_canonical_metadata_cache_invalidates_changed_episode_metadata(tmp_path):
    metadata_path = tmp_path / "episodes.parquet"
    metadata_path.write_bytes(b"version-one")
    cache_path = tmp_path / "index.pkl"
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset._metadata_index_cache_key = "unit-key"
    shard = SimpleNamespace(
        episode_metadata_path=metadata_path,
        episode_metadata_sha256=canonical._hash_file(metadata_path),
        episode_metadata_size=metadata_path.stat().st_size,
        episode_metadata_mtime_ns=metadata_path.stat().st_mtime_ns,
        episode_metadata_ctime_ns=metadata_path.stat().st_ctime_ns,
    )

    dataset._write_metadata_index_cache(cache_path, [shard])
    assert dataset._read_metadata_index_cache(cache_path) is not None

    metadata_path.write_bytes(b"version-two")
    assert dataset._read_metadata_index_cache(cache_path) is None


def test_realsource_subtask_segments_are_half_open_and_overlap_deterministic(
    tmp_path,
):
    segments_path = tmp_path / "subtask_segments.parquet"
    pd.DataFrame(
        [
            {
                "episode_index": 0,
                "segment_index": 0,
                "subtask_index": 1,
                "subtask": "coarse action",
                "start_frame": 5,
                "end_frame_exclusive": 15,
                "source_start_frame": 5,
                "source_end_frame": 15,
            },
            {
                "episode_index": 0,
                "segment_index": 1,
                "subtask_index": 2,
                "subtask": "later action",
                "start_frame": 8,
                "end_frame_exclusive": 12,
                "source_start_frame": 8,
                "source_end_frame": 12,
            },
            {
                "episode_index": 0,
                "segment_index": 2,
                "subtask_index": 3,
                "subtask": "shortest action",
                "start_frame": 8,
                "end_frame_exclusive": 10,
                "source_start_frame": 8,
                "source_end_frame": 10,
            },
            {
                "episode_index": 0,
                "segment_index": 3,
                "subtask_index": 4,
                "subtask": "empty annotation",
                "start_frame": 20,
                "end_frame_exclusive": 20,
                "source_start_frame": 20,
                "source_end_frame": 20,
            },
        ]
    ).to_parquet(segments_path)

    spans_by_episode, summary = canonical._load_subtask_segment_spans(
        segments_path,
        episode_lengths={0: 30},
    )
    assert summary["row_count"] == 4
    assert summary["usable_span_count"] == 3
    assert summary["zero_length_span_count"] == 1
    assert summary["unaligned_source_row_count"] == 0

    episode = canonical.EpisodeSpec(
        local_start=0,
        length=30,
        task="task",
        video_paths={},
        video_base_frames={},
        subtask_spans=spans_by_episode[0],
        episode_index=0,
    )
    dataset = canonical.CanonicalSubsetVLADataset.__new__(
        canonical.CanonicalSubsetVLADataset
    )
    assert dataset._subtask_label_for_window(episode, 4, None) is None
    assert (
        dataset._subtask_label_for_window(episode, 8, None)
        == "shortest action"
    )
    assert (
        dataset._subtask_label_for_window(episode, 10, None)
        == "later action"
    )
    assert (
        dataset._subtask_label_for_window(episode, 12, None)
        == "coarse action"
    )
    assert dataset._subtask_label_for_window(episode, 15, None) is None
    assert dataset._subtask_label_for_window(episode, 20, None) is None


def test_realsource_collect_mail_segment_alignment_drops_only_extra_annotation(
    tmp_path,
):
    segments_path = tmp_path / "subtask_segments.parquet"
    pd.DataFrame(
        [
            {
                "episode_index": 122,
                "segment_index": 0,
                "subtask_index": 1,
                "subtask": "extra annotation",
                "start_frame": 0,
                "end_frame_exclusive": 2,
            },
            {
                "episode_index": 123,
                "segment_index": 0,
                "subtask_index": 2,
                "subtask": "aligned data episode 122",
                "start_frame": 1,
                "end_frame_exclusive": 3,
            },
        ]
    ).to_parquet(segments_path)
    episode_lengths = {index: 10 for index in range(390)}
    source_map = canonical._realsource_subtask_episode_index_map(
        dataset_id=canonical.REALSOURCE_COLLECT_MAIL_DATASET_ID,
        episode_lengths=episode_lengths,
        annotation_alignment={
            "algorithm": (
                canonical.REALSOURCE_COLLECT_MAIL_ALIGNMENT_ALGORITHM
            )
        },
    )

    spans_by_episode, summary = canonical._load_subtask_segment_spans(
        segments_path,
        episode_lengths=episode_lengths,
        source_episode_index_map=source_map,
    )

    assert 121 not in spans_by_episode
    assert [span.label for span in spans_by_episode[122]] == [
        "aligned data episode 122"
    ]
    assert summary["unaligned_source_row_count"] == 1


def test_canonical_metadata_cache_invalidates_changed_subtask_segments(
    tmp_path,
):
    metadata_path = tmp_path / "episodes.parquet"
    metadata_path.write_bytes(b"episode-metadata")
    segments_path = tmp_path / "subtask_segments.parquet"
    segments_path.write_bytes(b"segments-v1")
    cache_path = tmp_path / "index.pkl"
    dataset = canonical.CanonicalSubsetVLADataset.__new__(
        canonical.CanonicalSubsetVLADataset
    )
    dataset._metadata_index_cache_key = "unit-key"
    dataset.append_subtask_to_prompt = True
    metadata_stat = metadata_path.stat()
    segments_stat = segments_path.stat()
    shard = SimpleNamespace(
        dataset_id=(
            "RealSourceData/RealSource-World/fixture"
        ),
        episode_metadata_path=metadata_path,
        episode_metadata_sha256=canonical._hash_file(metadata_path),
        episode_metadata_size=metadata_stat.st_size,
        episode_metadata_mtime_ns=metadata_stat.st_mtime_ns,
        episode_metadata_ctime_ns=metadata_stat.st_ctime_ns,
        subtask_segments_path=segments_path,
        subtask_segments_sha256=canonical._hash_file(segments_path),
        subtask_segments_size=segments_stat.st_size,
        subtask_segments_mtime_ns=segments_stat.st_mtime_ns,
        subtask_segments_ctime_ns=segments_stat.st_ctime_ns,
    )
    dataset._write_metadata_index_cache(cache_path, [shard])
    assert dataset._read_metadata_index_cache(cache_path) is not None

    segments_path.write_bytes(b"segments-v2")
    assert dataset._read_metadata_index_cache(cache_path) is None


def test_subtask_segments_survive_lazy_worker_serialization(tmp_path):
    segments_path = tmp_path / "subtask_segments.parquet"
    segments_path.write_bytes(b"authenticated-segments")
    span = canonical.SubtaskSpan(
        "pick object",
        start_frame=10,
        end_frame=20,
        subtask_index=3,
        segment_index=7,
        boundary_semantics="source_frame_half_open",
    )
    shard = SimpleNamespace(
        subtask_segments_path=segments_path,
        subtask_segments_sha256=canonical._hash_file(segments_path),
        episodes=[
            canonical.EpisodeSpec(
                local_start=0,
                length=30,
                task="task",
                video_paths={},
                video_base_frames={},
                subtask_spans=(span,),
                episode_index=4,
            )
        ],
    )

    restored = pickle.loads(pickle.dumps(shard))
    assert restored.subtask_segments_sha256 == canonical._hash_file(
        segments_path
    )
    assert restored.episodes[0].subtask_spans == (span,)


def test_canonical_provenance_binds_subtask_policy_and_episode_metadata(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    metadata_path = tmp_path / "episodes.parquet"
    metadata_path.write_bytes(b"episode-metadata")
    segments_path = tmp_path / "subtask_segments.parquet"
    segments_path.write_bytes(b"subtask-segments")
    episode = SimpleNamespace(
        subtask_spans=(canonical.SubtaskSpan("grasp chain"),)
    )
    shard = SimpleNamespace(
        dataset_id="dataset",
        sid="sid",
        revision="main",
        data_relative_path="data/chunk-000/file-000.parquet",
        episode_metadata_path=metadata_path,
        episode_metadata_sha256=canonical._hash_file(metadata_path),
        subtask_segments_path=segments_path,
        subtask_segments_sha256=canonical._hash_file(segments_path),
        episodes=[episode],
    )
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.data_cfg = {
        "append_subtask_to_prompt": True,
        "subtask_prompt_source_column": "subtask_index",
        "subtask_prompt_label_column": "local_subtask_text",
        "subtask_prompt_append_probability": 1.0,
        "subtask_prompt_separator": " | ",
        "subtask_prompt_ignored_labels": ["__unlabeled__"],
    }
    dataset.append_subtask_to_prompt = True
    dataset.subtask_prompt_source_column = "subtask_index"
    dataset.subtask_prompt_label_column = "local_subtask_text"
    dataset.manifest_path = manifest_path
    dataset._metadata_index_cache_key = "cache-key"
    dataset.shards = [shard]

    provenance = dataset.dataset_provenance()
    assert provenance["subtask_prompt"]["selected_usable_span_count"] == 1
    assert provenance["episode_metadata_sources"] == [
        {
            "path": metadata_path.as_posix(),
            "sha256": canonical._hash_file(metadata_path),
        }
    ]
    assert provenance["subtask_segment_sources"] == [
        {
            "path": segments_path.as_posix(),
            "sha256": canonical._hash_file(segments_path),
            "schema": canonical.SUBTASK_SEGMENTS_SCHEMA,
            "frame_coordinates": "raw_source_frames",
            "boundary_semantics": (
                "start_inclusive_end_exclusive"
            ),
            "overlap_resolution": (
                canonical.SUBTASK_OVERLAP_RESOLUTION
            ),
        }
    ]

    provenance_path = tmp_path / "dataset_provenance.json"
    dataset.save_dataset_provenance(provenance_path)
    original_bytes = provenance_path.read_bytes()
    dataset.save_dataset_provenance(provenance_path)
    shard.episode_metadata_sha256 = "changed"
    with pytest.raises(ValueError, match="immutable resume binding"):
        dataset.save_dataset_provenance(provenance_path)
    assert provenance_path.read_bytes() == original_bytes


def test_canonical_sample_returns_qwen_frames_without_duplicate_qwen_views(tmp_path):
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.video_target_shift_steps = 1
    dataset._compact_offsets_cache = np.asarray([-1, 0, 1], dtype=np.int64)
    dataset.append_subtask_to_prompt = True
    dataset.data_cfg = {
        "append_subtask_to_prompt": True,
        "subtask_prompt_source_column": "subtask_index",
        "subtask_prompt_label_column": "local_subtask_text",
        "subtask_prompt_append_probability": 1.0,
        "subtask_prompt_separator": " | ",
        "subtask_prompt_ignored_labels": ["__unlabeled__"],
    }

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
            canonical.SubtaskSpan(
                "grasp object",
                start_frame=2,
                end_frame=3,
                subtask_index=2,
                segment_index=1,
                boundary_semantics="source_frame_half_open",
            ),
        ),
    )
    context = {
        "shard_data": shard_data,
        "shard": shard,
        "episode": episode,
        "row_base": 1,
        # A frozen 20 Hz eval row may map to a different raw 30 Hz source
        # frame. Subtask lookup must use this source coordinate.
        "source_base_index": 2,
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


def test_canonical_subtask_prompt_probability_zero_leaves_language_unchanged():
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.append_subtask_to_prompt = True
    dataset.data_cfg = {
        "append_subtask_to_prompt": True,
        "subtask_prompt_append_probability": 0.0,
        "subtask_prompt_separator": " | ",
        "subtask_prompt_ignored_labels": ["__unlabeled__"],
    }

    assert dataset._language_with_subtask("task", "grasp object") == "task"


def test_canonical_subtask_prompt_ignores_configured_unlabeled_value():
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.append_subtask_to_prompt = True
    dataset.data_cfg = {
        "append_subtask_to_prompt": True,
        "subtask_prompt_append_probability": 1.0,
        "subtask_prompt_separator": " | ",
        "subtask_prompt_ignored_labels": ["__unlabeled__"],
    }

    assert dataset._language_with_subtask("task", "__unlabeled__") == "task"


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


def test_canonical_sample_context_clamps_action_rows_to_sidecar_length(monkeypatch):
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.index_windows_lazily = False
    dataset.windows = [canonical.WindowSpec(shard_index=0, episode_index=0, base_index=5)]
    dataset.data_file_prefetch_shards = 0
    dataset._action_offsets = np.arange(50, dtype=np.int64)
    dataset.video_horizon = 8
    dataset.video_target_shift_steps = 0
    dataset.video_frame_stride = 1
    dataset._compact_offsets_cache = None

    shard = SimpleNamespace(
        sid="sid",
        data_relative_path="data/chunk-000/file-000.parquet",
        decode_camera_slots=(),
        vjepa_camera_slots=(),
        qwen_camera_slots=(),
        episodes=[
            canonical.EpisodeSpec(
                local_start=0,
                length=20,
                task="task",
                video_paths={},
                video_base_frames={},
            )
        ],
    )
    shard_data = SimpleNamespace(
        state=np.zeros((10, canonical.STATE_DIM), dtype=np.float32),
        action=np.zeros((10, canonical.ACTION_DIM), dtype=np.float32),
        action_mask=np.ones((10, canonical.ACTION_DIM), dtype=bool),
        timestamp=np.zeros(10, dtype=np.float32),
        frame_index=np.arange(10, dtype=np.int64),
        episode_index=np.zeros(10, dtype=np.int64),
        task_index=np.zeros(10, dtype=np.int64),
    )
    dataset.shards = [shard]
    monkeypatch.setattr(dataset, "_schedule_shard_data_prefetch", lambda _shard_index: None)
    monkeypatch.setattr(dataset, "_get_shard_data", lambda _shard_index: shard_data)

    context = dataset._sample_context(0)

    assert context["row_base"] == 5
    assert context["action_rows"][-1] == 9
    assert context["action_rows"].max() == 9
    np.testing.assert_array_equal(
        context["action_is_pad"],
        np.asarray([False, False, False, False, False] + [True] * 45, dtype=bool),
    )


def test_canonical_getitems_retries_sidecar_tail_window(monkeypatch):
    dataset = canonical.CanonicalSubsetVLADataset.__new__(canonical.CanonicalSubsetVLADataset)
    dataset.skip_corrupt_videos = True
    dataset.max_sample_decode_retries = 2
    dataset.total_windows = 1_000_000
    dataset._bad_video_paths = set()
    dataset._bad_video_warning_count = 0
    dataset.pyav_corrupt_warning_limit = 10

    attempts = []

    def fake_sample_context(index):
        attempts.append(index)
        if index in {1, 2}:
            raise canonical._RecoverableSampleError("sidecar tail")
        return {"sample_index": index, "video_frames": {}}

    monkeypatch.setattr(dataset, "_sample_context", fake_sample_context)
    monkeypatch.setattr(
        dataset,
        "_sample_from_context",
        lambda context, decoded_frames=None: {"sample_index": context["sample_index"]},
    )

    samples = dataset.__getitems__([1, 2])

    assert attempts == [1, dataset._retry_index(1, 1), dataset._retry_index(2, 1)]
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
