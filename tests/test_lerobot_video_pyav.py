import numpy as np
import av

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps


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
