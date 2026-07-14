import numpy as np

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class _RecordingPolicy:
    def __init__(self):
        self.kwargs = None

    def predict_action(self, **kwargs):
        self.kwargs = kwargs
        return {"normalized_actions": np.zeros((1, 2, 3), dtype=np.float32)}


def test_websocket_server_preserves_qwen_tensor_frames():
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy)
    frames = np.zeros((1, 3, 384, 384, 3), dtype=np.uint8)

    response = server._route_message(
        {
            "type": "infer",
            "payload": {
                "qwen_frames": frames,
                "instructions": ["test task"],
                "state": np.zeros((1, 1, 19), dtype=np.float32),
            },
        }
    )

    assert response["ok"] is True
    assert policy.kwargs["qwen_frames"] is frames
    assert policy.kwargs["qwen_frames"].dtype == np.uint8


def test_websocket_server_keeps_legacy_batch_images_compatible():
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy)

    response = server._route_message(
        {
            "type": "infer",
            "payload": {
                "batch_images": [[np.zeros((8, 8, 3), dtype=np.uint8)]],
                "instructions": ["test task"],
            },
        }
    )

    assert response["ok"] is True
    assert policy.kwargs["batch_images"][0][0].mode == "RGB"


def test_websocket_server_rejects_ambiguous_image_payload():
    server = WebsocketPolicyServer(_RecordingPolicy())

    response = server._route_message(
        {
            "type": "infer",
            "payload": {
                "batch_images": [[]],
                "qwen_frames": np.zeros((1, 3, 8, 8, 3), dtype=np.uint8),
            },
        }
    )

    assert response["ok"] is False
    assert "must not include both" in response["error"]["message"]


def test_websocket_server_requires_an_image_payload():
    server = WebsocketPolicyServer(_RecordingPolicy())

    response = server._route_message({"type": "infer", "payload": {"instructions": ["test"]}})

    assert response["ok"] is False
    assert "batch_images" in response["error"]["message"]
    assert "qwen_frames" in response["error"]["message"]
