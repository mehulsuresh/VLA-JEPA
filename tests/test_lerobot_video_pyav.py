import numpy as np
import av

from starVLA.dataloader.gr00t_lerobot.video import (
    _decode_pyav_nearest_frames,
    get_all_frames,
    get_frames_by_timestamps,
)


def _write_test_video(path, frame_count=5, fps=10):
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            frame_data = np.full((stream.height, stream.width, 3), index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def test_pyav_video_backend_reads_all_frames(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path)

    frames = get_all_frames(
        video_path.as_posix(),
        video_backend="pyav",
        video_backend_kwargs={"num_threads": 1},
    )

    assert frames.shape == (5, 16, 16, 3)


def test_pyav_video_backend_reads_timestamp_samples(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path)

    frames = get_frames_by_timestamps(
        video_path.as_posix(),
        [0.0, 0.2, 0.4],
        video_backend="pyav",
        video_backend_kwargs={"num_threads": 1},
    )

    assert frames.shape == (3, 16, 16, 3)
    assert frames[0].mean() < frames[1].mean() < frames[2].mean()


def test_pyav_timestamp_samples_preserve_order_duplicates_and_end_padding(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path)

    frames = get_frames_by_timestamps(
        video_path.as_posix(),
        [0.4, 0.2, 0.2, 100.0],
        video_backend="pyav",
        video_backend_kwargs={"num_threads": 1},
    )

    means = [float(frame.mean()) for frame in frames]
    assert frames.shape == (4, 16, 16, 3)
    assert means[0] > means[1]
    assert means[1] == means[2]
    assert means[3] == means[0]


def test_pyav_nearest_frame_selection_converts_only_selected_frames():
    class FakeFrame:
        def __init__(self, timestamp, value):
            self.time = timestamp
            self.pts = None
            self.value = value
            self.conversions = 0

        def to_ndarray(self, format):
            assert format == "rgb24"
            self.conversions += 1
            return np.full((2, 2, 3), self.value, dtype=np.uint8)

    class FakeContainer:
        def __init__(self, frames):
            self.frames = frames
            self.decoded = 0

        def decode(self, stream):
            del stream
            for frame in self.frames:
                self.decoded += 1
                yield frame

    frames = [FakeFrame(index / 10.0, index) for index in range(10)]
    container = FakeContainer(frames)

    selected = _decode_pyav_nearest_frames(
        container,
        stream=object(),
        requested_timestamps=np.asarray([0.5, 0.2, 0.2]),
        min_timestamp=0.0,
    )

    assert [int(frame[0, 0, 0]) for frame in selected] == [5, 2, 2]
    assert sum(frame.conversions for frame in frames) == 2
    assert container.decoded == 6
