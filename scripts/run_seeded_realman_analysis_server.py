#!/usr/bin/env python3
"""Analysis-only policy server with request-scoped deterministic action noise."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

from deployment.model_server.server_policy import build_argparser, main
from starVLA.model.framework.VLA_JEPA import VLA_JEPA


_original_predict_action = VLA_JEPA.predict_action


def _seeded_predict_action(self, *args, **kwargs):
    inference_seed = kwargs.pop("inference_seed", None)
    if inference_seed is not None:
        seed = int(inference_seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return _original_predict_action(self, *args, **kwargs)


VLA_JEPA.predict_action = _seeded_predict_action


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
