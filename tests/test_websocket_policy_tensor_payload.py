import numpy as np
import pytest

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class _RecordingPolicy:
    def __init__(self):
        self.kwargs = None

    def predict_action(self, **kwargs):
        self.kwargs = kwargs
        return {"normalized_actions": np.zeros((1, 2, 3), dtype=np.float32)}


RTC_METADATA = {
    "action_dim": 18,
    "action_horizon": 50,
    "rtc_inference_contract": {
        "training_enabled": True,
        "method": "prefix",
        "max_prefix_len": 10,
    },
}


def test_websocket_server_preserves_qwen_tensor_frames():
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy, metadata=RTC_METADATA)
    frames = np.zeros((1, 3, 384, 384, 3), dtype=np.uint8)
    prev_actions = np.zeros((1, 5, 18), dtype=np.float32)

    response = server._route_message(
        {
            "type": "infer",
            "payload": {
                "qwen_frames": frames,
                "instructions": ["test task"],
                "state": np.zeros((1, 1, 19), dtype=np.float32),
                "prev_actions": prev_actions,
                "prefix_len": 5,
                "rtc_config": {"enabled": True, "method": "prefix"},
            },
        }
    )

    assert response["ok"] is True
    assert policy.kwargs["qwen_frames"] is frames
    assert policy.kwargs["qwen_frames"].dtype == np.uint8
    assert policy.kwargs["prev_actions"] is prev_actions
    assert policy.kwargs["prefix_len"] == 5
    assert policy.kwargs["rtc_config"] == {"enabled": True, "method": "prefix"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prefix_len": 0}, "prefix_len must be in"),
        ({"prefix_len": 11, "prev_actions": np.zeros((1, 11, 18), dtype=np.float32)}, "prefix_len must be in"),
        ({"prefix_len": True}, "prefix_len must be an integer"),
        ({"rtc_config": {"enabled": False, "method": "prefix"}}, "enabled=true"),
        ({"rtc_config": {"enabled": True, "method": "suffix"}}, "method must be"),
        ({"prev_actions": np.zeros((5, 18), dtype=np.float32)}, "rank 3"),
        ({"prev_actions": np.zeros((1, 4, 18), dtype=np.float32)}, "at least 5 rows"),
        ({"prev_actions": np.zeros((1, 51, 18), dtype=np.float32)}, "cannot exceed action_horizon"),
        ({"prev_actions": np.zeros((1, 5, 17), dtype=np.float32)}, "action dimension"),
        ({"prev_actions": np.full((1, 5, 18), np.nan, dtype=np.float32)}, "non-finite"),
    ],
)
def test_websocket_server_rejects_invalid_rtc_contract(overrides, message):
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy, metadata=RTC_METADATA)
    payload = {
        "qwen_frames": np.zeros((1, 3, 8, 8, 3), dtype=np.uint8),
        "instructions": ["test task"],
        "state": np.zeros((1, 1, 19), dtype=np.float32),
        "prev_actions": np.zeros((1, 5, 18), dtype=np.float32),
        "prefix_len": 5,
        "rtc_config": {"enabled": True, "method": "prefix"},
        **overrides,
    }

    response = server._route_message({"type": "infer", "payload": payload})

    assert response["ok"] is False
    assert message in response["error"]["message"]
    assert policy.kwargs is None


@pytest.mark.parametrize("missing", ["prev_actions", "prefix_len", "rtc_config"])
def test_websocket_server_requires_complete_rtc_payload(missing):
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(policy, metadata=RTC_METADATA)
    payload = {
        "qwen_frames": np.zeros((1, 3, 8, 8, 3), dtype=np.uint8),
        "instructions": ["test task"],
        "prev_actions": np.zeros((1, 5, 18), dtype=np.float32),
        "prefix_len": 5,
        "rtc_config": {"enabled": True, "method": "prefix"},
    }
    del payload[missing]

    response = server._route_message({"type": "infer", "payload": payload})

    assert response["ok"] is False
    assert "must include prev_actions, prefix_len, and rtc_config" in response["error"]["message"]
    assert policy.kwargs is None


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


def test_realman_tensor_contract_rejects_legacy_batch_images():
    policy = _RecordingPolicy()
    server = WebsocketPolicyServer(
        policy,
        metadata={"realman_input_contract": {"payload_key": "qwen_frames"}},
    )

    response = server._route_message(
        {
            "type": "infer",
            "payload": {
                "batch_images": [[np.zeros((8, 8, 3), dtype=np.uint8)]],
                "instructions": ["test task"],
            },
        }
    )

    assert response["ok"] is False
    assert "requires `qwen_frames`" in response["error"]["message"]
    assert policy.kwargs is None


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
