from pathlib import Path

import torch
from safetensors.torch import save_file

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def test_load_pretrained_backbones_loads_full_safetensors_checkpoint(
    tmp_path: Path,
) -> None:
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    expected = {
        name: tensor.detach().clone() for name, tensor in source.state_dict().items()
    }
    checkpoint = tmp_path / "model.safetensors"
    save_file(expected, checkpoint)

    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()

    loaded = TrainerUtils.load_pretrained_backbones(target, checkpoint)

    assert loaded is target
    for name, actual in target.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0.0, atol=0.0)
