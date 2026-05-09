from pathlib import Path
from types import SimpleNamespace

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
