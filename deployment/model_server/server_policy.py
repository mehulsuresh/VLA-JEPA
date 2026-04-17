# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
import os

import torch

from deployment.model_server.checkpoint_utils import build_policy_metadata, resolve_policy_checkpoint
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    resolved_ckpt_path = resolve_policy_checkpoint(args.ckpt_path)
    logging.info("Loading policy checkpoint from `%s`", resolved_ckpt_path)

    vla = baseframework.from_pretrained(str(resolved_ckpt_path))  # TODO should auto detect framework from model path

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.cuda)}")
    else:
        device = torch.device("cpu")
        if args.use_bf16:
            logging.warning("Ignoring --use_bf16 because CUDA is not available")

    if args.use_bf16 and device.type == "cuda":
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    metadata = build_policy_metadata(vla, resolved_ckpt_path)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", type=int, default=0)
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10091))
    logging.info("Waiting for VSCode attach on 0.0.0.0:10091")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


def env_flag_enabled(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if env_flag_enabled("DEBUG"):
        logging.info("DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
