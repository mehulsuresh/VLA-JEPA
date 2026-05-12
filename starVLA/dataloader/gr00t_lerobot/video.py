# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import av
import cv2
import numpy as np
from functools import lru_cache

import torch  # noqa: F401 # isort: skip
import torchvision  # noqa: F401 # isort: skip

# Import decord with graceful fallback
try:
    import decord  # noqa: F401

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


def _freeze_backend_kwargs(video_backend_kwargs: dict | None) -> tuple[tuple[str, object], ...]:
    if not video_backend_kwargs:
        return ()
    return tuple(sorted(video_backend_kwargs.items()))


def _configure_pyav_stream(container: av.container.InputContainer, video_backend_kwargs: dict | None):
    stream = container.streams.video[0]
    thread_count = int((video_backend_kwargs or {}).get("num_threads", 1))
    if thread_count > 0:
        stream.codec_context.thread_count = thread_count
    return stream


def _pyav_frame_time_seconds(frame: av.VideoFrame, stream, frame_index: int) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    if stream.average_rate:
        return float(frame_index / float(stream.average_rate))
    return float(frame_index)


@lru_cache(maxsize=512)
def _get_decord_frame_timestamps(
    video_path: str,
    frozen_backend_kwargs: tuple[tuple[str, object], ...],
) -> np.ndarray:
    vr = decord.VideoReader(video_path, **dict(frozen_backend_kwargs))
    num_frames = len(vr)
    return vr.get_frame_timestamp(range(num_frames))


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_at(indices=indices).data.numpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frame_ts = _get_decord_frame_timestamps(
            video_path,
            _freeze_backend_kwargs(video_backend_kwargs),
        )
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_played_at(seconds=timestamps).data.numpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        torchvision.set_video_backend("pyav")
        loaded_frames = []
        loaded_ts = []
        
        reader = None
        try:
            reader = torchvision.io.VideoReader(video_path, "video")
            
            for target_ts in timestamps:
                # Reset reader state
                reader.seek(target_ts, keyframes_only=True)
                
                closest_frame = None
                closest_ts_diff = float('inf')
                
                for frame in reader:
                    current_ts = frame["pts"]
                    current_diff = abs(current_ts - target_ts)
                    
                    if closest_frame is None:
                        closest_frame = frame
                    
                    if current_diff < closest_ts_diff:
                        # Release previous frame reference
                        if closest_frame is not None:
                            del closest_frame
                        closest_ts_diff = current_diff
                        closest_frame = frame
                    else:
                        # Difference started growing, stop search
                        break
                
                if closest_frame is not None:
                    frame_data = closest_frame["data"]
                    if isinstance(frame_data, torch.Tensor):
                        frame_data = frame_data.cpu().numpy()
                    loaded_frames.append(frame_data)
                    loaded_ts.append(closest_frame["pts"])
                    
                    # Immediately release frame reference
                    del closest_frame
                    
        finally:
            # Thoroughly clean resources
            if reader is not None:
                if hasattr(reader, '_c'):
                    reader._c = None
                if hasattr(reader, 'container'):
                    reader.container.close()
                    reader.container = None
            # Force garbage collection
            import gc
            gc.collect()
        
        frames = np.array(loaded_frames)
        return frames.transpose(0, 2, 3, 1)
    elif video_backend == "pyav":
        requested_ts = np.asarray(timestamps, dtype=np.float64)
        if requested_ts.ndim == 0:
            requested_ts = requested_ts[None]
        if requested_ts.size == 0:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)

        container = av.open(video_path)
        try:
            stream = _configure_pyav_stream(container, video_backend_kwargs)
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            stop_after = float(np.max(requested_ts)) + (2.0 / fps if fps > 0 else 1.0)
            frame_ts = []
            frames = []
            for frame_index, frame in enumerate(container.decode(stream)):
                ts = _pyav_frame_time_seconds(frame, stream, frame_index)
                frame_ts.append(ts)
                frames.append(frame.to_ndarray(format="rgb24"))
                if ts > stop_after:
                    break
        finally:
            container.close()

        if not frames:
            raise ValueError(f"Unable to read frames from video file: {video_path}")
        closest_indices = np.abs(np.asarray(frame_ts)[:, None] - requested_ts[None, :]).argmin(axis=0)
        return np.stack([frames[int(index)] for index in closest_indices], axis=0)
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        frames = decoder.get_frames_at(indices=range(len(decoder)))
        return frames.data.numpy(), frames.pts_seconds.numpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        try:
            stream = _configure_pyav_stream(container, video_backend_kwargs)
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
            frames = np.array(frames)
        finally:
            container.close()
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"].numpy())
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames
