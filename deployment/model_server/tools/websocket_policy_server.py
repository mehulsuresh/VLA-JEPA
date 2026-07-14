# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import traceback

import numpy as np
import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy
from . import image_tools

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        output_logger=None,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_logger = output_logger
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def _validate_rtc_payload(self, payload: dict) -> str | None:
        """Validate the opt-in RTC prefix against the checkpoint contract."""

        rtc_keys = ("prev_actions", "prefix_len", "rtc_config")
        present = [key for key in rtc_keys if key in payload]
        if not present:
            return None

        contract = self._metadata.get("rtc_inference_contract")
        if not isinstance(contract, dict):
            return "Payload requested RTC, but the policy metadata does not advertise an RTC contract"
        if not bool(contract.get("training_enabled")):
            return "Payload requested RTC, but RTC was not enabled for this checkpoint's training"
        missing = [key for key in rtc_keys if key not in payload]
        if missing:
            return f"RTC payload must include prev_actions, prefix_len, and rtc_config; missing {missing}"

        rtc_config = payload.get("rtc_config")
        if not isinstance(rtc_config, dict) or rtc_config.get("enabled") is not True:
            return "rtc_config must be a dict with enabled=true"
        expected_method = str(contract.get("method", "prefix"))
        if str(rtc_config.get("method", "")) != expected_method:
            return f"rtc_config.method must be {expected_method!r}"

        prefix_len = payload.get("prefix_len")
        if isinstance(prefix_len, (bool, np.bool_)) or not isinstance(
            prefix_len, (int, np.integer)
        ):
            return "prefix_len must be an integer"
        prefix_len = int(prefix_len)
        max_prefix_len = int(contract.get("max_prefix_len", 0) or 0)
        if prefix_len < 1 or prefix_len > max_prefix_len:
            return f"prefix_len must be in [1, {max_prefix_len}], got {prefix_len}"

        try:
            prev_actions = np.asarray(payload.get("prev_actions"))
        except Exception:
            return "prev_actions must be a numeric [1, prefix_len, action_dim] array"
        expected_action_dim = self._metadata.get("action_dim")
        expected_horizon = self._metadata.get("action_horizon")
        expected_shape = (
            1,
            prefix_len,
            int(expected_action_dim) if expected_action_dim is not None else None,
        )
        if prev_actions.ndim != 3:
            return f"prev_actions must have rank 3, got shape {prev_actions.shape}"
        if prev_actions.shape[0] != 1 or prev_actions.shape[1] < prefix_len:
            return (
                "prev_actions must have shape [1, at_least_prefix_len, action_dim]; "
                f"expected batch 1 and at least {prefix_len} rows, got {prev_actions.shape}"
            )
        if expected_horizon is not None and prev_actions.shape[1] > int(expected_horizon):
            return (
                f"prev_actions cannot exceed action_horizon={int(expected_horizon)}, "
                f"got {prev_actions.shape[1]} rows"
            )
        if expected_shape[2] is not None and prev_actions.shape[2] != expected_shape[2]:
            return (
                f"prev_actions action dimension must be {expected_shape[2]}, "
                f"got {prev_actions.shape[2]}"
            )
        if prev_actions.dtype.kind not in "fiu":
            return f"prev_actions must be numeric, got dtype {prev_actions.dtype}"
        if not np.isfinite(prev_actions).all():
            return "prev_actions contains non-finite values"
        return None

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                ret = self._route_message(msg, remote_address=websocket.remote_address)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logging.exception("Unhandled websocket server error for %s", websocket.remote_address)
                error_payload = {
                    "status": "error",
                    "ok": False,
                    "type": "internal_error",
                    "request_id": "unknown",
                    "error": {
                        "message": "Internal server error",
                        "traceback": traceback.format_exc(),
                    },
                }
                await websocket.send(packer.pack(error_payload))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Details sent in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: dict, remote_address=None) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Always returns a dict containing:
            {
              "status": "ok" | "error",
              "ok": bool,
              "type": <str>,
              "request_id": <str>,
              ... (data | error)
            }
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        if not isinstance(msg, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "invalid_request",
                "request_id": "unknown",
                "error": {"message": f"Expected dict message, got {type(msg).__name__}"},
            }

        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer
        payload = msg.get("payload", msg)         # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer
        elif mtype == "infer":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }
            image_payload_keys = [key for key in ("batch_images", "qwen_frames") if key in payload]
            if not image_payload_keys:
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must include `batch_images` or `qwen_frames`"},
                }
            if len(image_payload_keys) != 1:
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must not include both `batch_images` and `qwen_frames`"},
                }
            input_contract = self._metadata.get("realman_input_contract")
            required_image_key = input_contract.get("payload_key") if isinstance(input_contract, dict) else None
            if required_image_key and image_payload_keys[0] != required_image_key:
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": (
                            f"Policy input contract requires `{required_image_key}`; "
                            f"received `{image_payload_keys[0]}`. Refusing an incompatible "
                            "image preprocessing path."
                        )
                    },
                }
            rtc_error = self._validate_rtc_payload(payload)
            if rtc_error is not None:
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": rtc_error},
                }
            try:
                infer_payload = dict(payload)
                if "batch_images" in infer_payload:
                    infer_payload["batch_images"] = image_tools.to_pil_preserve(infer_payload["batch_images"])
                output_dict = self._policy.predict_action(**infer_payload)
                if self._output_logger is not None:
                    try:
                        self._output_logger(
                            request_id=req_id,
                            remote_address=remote_address,
                            payload=infer_payload,
                            output=output_dict,
                        )
                    except Exception:
                        logging.exception("Policy output logging failed (request_id=%s)", req_id)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                    },
                }
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": output_dict,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
