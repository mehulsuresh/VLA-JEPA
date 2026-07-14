import numpy as np

from starVLA.model.framework.VLA_JEPA import VLA_JEPA


def test_qwen_frame_conversion_copies_read_only_msgpack_style_array(recwarn):
    raw = bytes(3 * 8 * 8 * 3)
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(3, 8, 8, 3)
    assert frames.flags.writeable is False
    model = VLA_JEPA.__new__(VLA_JEPA)

    tensor = model._qwen_frame_batch_to_chw_tensor(frames)

    assert tuple(tensor.shape) == (3, 3, 8, 8)
    assert not recwarn.list
