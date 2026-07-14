import argparse
import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "scripts/probe_qwen35_fast_linear_attention.py"
DOCKER_BUILD_PATH = REPO_ROOT / "scripts/docker_build_training.sh"
IMAGE_BUILD_PATH = REPO_ROOT / "scripts/build_magna_a100_image.sh"
DOCKERFILE_PATH = REPO_ROOT / "docker/Dockerfile.py313"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("qwen35_fast_linear_attention_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_docker_build(
    tmp_path: Path,
    extra_env: dict[str, str] | None = None,
    script_args: tuple[str, ...] = (),
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "docker_args.txt"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': > "${DOCKER_CAPTURE_PATH}"\n'
        'printf "%s\\n" "$@" >> "${DOCKER_CAPTURE_PATH}"\n',
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "DOCKER_CAPTURE_PATH": str(capture_path),
        }
    )
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(DOCKER_BUILD_PATH), *script_args],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    args = capture_path.read_text(encoding="utf-8").splitlines() if capture_path.exists() else []
    return result, args


def _build_arg_values(args: list[str]) -> dict[str, str]:
    values = {}
    for index, arg in enumerate(args):
        if arg != "--build-arg":
            continue
        name, value = args[index + 1].split("=", 1)
        values[name] = value
    return values


def test_probe_contract_matches_qwen35_2b_and_sm90_defaults():
    probe = _load_probe_module()

    assert probe.DEFAULT_FLA_VERSION == "0.5.1"
    assert probe.DEFAULT_CAUSAL_CONV1D_VERSION == "1.6.2.post1"
    assert probe.DEFAULT_TRANSFORMERS_VERSION == "5.13.1"
    assert probe.DEFAULT_TILELANG_VERSION == "0.1.9"
    assert probe.DEFAULT_TVM_FFI_VERSION == "0.1.10"
    assert probe.QWEN35_2B_LINEAR_ATTN_SHAPE == {
        "hidden_size": 2048,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
    }
    assert probe._build_parser().parse_args([]).expected_compute_capability == (9, 0)


@pytest.mark.parametrize("value", ["9", "sm90", "9.x", "9.0.0", ""])
def test_probe_rejects_ambiguous_compute_capability(value):
    probe = _load_probe_module()

    with pytest.raises(argparse.ArgumentTypeError):
        probe._parse_compute_capability(value)


def test_probe_requires_every_transformers_fast_path_binding():
    probe = _load_probe_module()
    complete = SimpleNamespace(**{name: (lambda: None) for name in probe.REQUIRED_MODELING_BINDINGS})

    assert set(probe._require_modeling_bindings(complete)) == set(probe.REQUIRED_MODELING_BINDINGS)
    complete.causal_conv1d_update = None
    with pytest.raises(RuntimeError, match="causal_conv1d_update"):
        probe._require_modeling_bindings(complete)


def test_probe_fails_closed_on_transformers_version_mismatch(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda _distribution: "5.14.0")

    with pytest.raises(RuntimeError, match="expected '5.13.1', found '5.14.0'"):
        probe._require_exact_package_version("transformers", "5.13.1")


def test_probe_help_does_not_require_ml_runtime_imports():
    result = subprocess.run(
        ["python", str(PROBE_PATH), "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--imports-only" in result.stdout
    assert "--expected-compute-capability" in result.stdout


def test_generic_image_build_keeps_fast_linear_attention_off_by_default(tmp_path):
    result, args = _fake_docker_build(tmp_path)

    assert result.returncode == 0, result.stderr
    build_args = _build_arg_values(args)
    assert build_args["INSTALL_FAST_LINEAR_ATTN"] == "0"
    assert build_args["FAST_LINEAR_ATTN_SPEC"] == "flash-linear-attention[cuda]==0.5.1"
    assert build_args["CAUSAL_CONV1D_SPEC"] == "causal-conv1d==1.6.2.post1"
    assert build_args["FAST_LINEAR_ATTN_TRANSFORMERS_SPEC"] == "transformers==5.13.1"
    assert build_args["FAST_LINEAR_ATTN_TILELANG_SPEC"] == "tilelang==0.1.9"
    assert (
        build_args["FAST_LINEAR_ATTN_TVM_FFI_SPEC"]
        == "apache-tvm-ffi==0.1.10"
    )
    assert build_args["FAST_LINEAR_ATTN_CUDA_ARCH_LIST"] == "8.0"


def test_generic_image_build_forwards_explicit_sm90_fast_path(tmp_path):
    result, args = _fake_docker_build(
        tmp_path,
        {
            "INSTALL_FAST_LINEAR_ATTN": "1",
            "FAST_LINEAR_ATTN_CUDA_ARCH_LIST": "9.0",
            "FAST_LINEAR_ATTN_MAX_JOBS": "47",
        },
    )

    assert result.returncode == 0, result.stderr
    build_args = _build_arg_values(args)
    assert build_args["INSTALL_FAST_LINEAR_ATTN"] == "1"
    assert build_args["FAST_LINEAR_ATTN_CUDA_ARCH_LIST"] == "9.0"
    assert build_args["FAST_LINEAR_ATTN_MAX_JOBS"] == "47"


def test_managed_environment_overrides_duplicate_raw_build_args(tmp_path):
    result, args = _fake_docker_build(
        tmp_path,
        script_args=("--build-arg", "INSTALL_FAST_LINEAR_ATTN=1"),
    )

    assert result.returncode == 0, result.stderr
    assert _build_arg_values(args)["INSTALL_FAST_LINEAR_ATTN"] == "0"


@pytest.mark.parametrize(
    ("extra_env", "message"),
    [
        ({"INSTALL_FAST_LINEAR_ATTN": "yes"}, "must be 0 or 1"),
        (
            {"INSTALL_FAST_LINEAR_ATTN": "1", "FAST_LINEAR_ATTN_SPEC": "flash-linear-attention"},
            "must be an exact",
        ),
        (
            {"INSTALL_FAST_LINEAR_ATTN": "1", "CAUSAL_CONV1D_SPEC": "causal-conv1d>=1.4"},
            "must be an exact",
        ),
        (
            {"INSTALL_FAST_LINEAR_ATTN": "1", "FAST_LINEAR_ATTN_TRANSFORMERS_SPEC": "transformers"},
            "must be an exact",
        ),
        (
            {
                "INSTALL_FAST_LINEAR_ATTN": "1",
                "FAST_LINEAR_ATTN_TILELANG_SPEC": "tilelang>=0.1.9",
            },
            "must be an exact",
        ),
        (
            {
                "INSTALL_FAST_LINEAR_ATTN": "1",
                "FAST_LINEAR_ATTN_TVM_FFI_SPEC": "apache-tvm-ffi>=0.1.10",
            },
            "must be an exact",
        ),
        (
            {"INSTALL_FAST_LINEAR_ATTN": "1", "FAST_LINEAR_ATTN_CUDA_ARCH_LIST": "native"},
            "explicit numeric CUDA architectures",
        ),
        (
            {"INSTALL_FAST_LINEAR_ATTN": "1", "FAST_LINEAR_ATTN_MAX_JOBS": "0"},
            "must be a positive integer",
        ),
    ],
)
def test_generic_image_build_fails_closed_before_docker(tmp_path, extra_env, message):
    result, args = _fake_docker_build(tmp_path, extra_env)

    assert result.returncode == 2
    assert message in result.stderr
    assert args == []


def test_dockerfile_forces_source_build_and_import_probe_when_enabled():
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    wrapper = IMAGE_BUILD_PATH.read_text(encoding="utf-8")

    assert "CAUSAL_CONV1D_FORCE_BUILD=TRUE" in dockerfile
    assert "PIP_NO_BINARY=causal-conv1d" in dockerfile
    assert "probe_qwen35_fast_linear_attention.py" in dockerfile
    assert "--imports-only" in dockerfile
    assert '"${FAST_LINEAR_ATTN_TVM_FFI_SPEC}" "${FAST_LINEAR_ATTN_TILELANG_SPEC}"' in dockerfile
    assert "python -m pip check" in dockerfile
    assert '--expected-transformers-version "${FAST_LINEAR_ATTN_TRANSFORMERS_SPEC##*==}"' in dockerfile
    assert 'INSTALL_FAST_LINEAR_ATTN="${INSTALL_FAST_LINEAR_ATTN:-0}"' in wrapper
    assert 'FAST_LINEAR_ATTN_TRANSFORMERS_SPEC="${FAST_LINEAR_ATTN_TRANSFORMERS_SPEC:-transformers==5.13.1}"' in wrapper
    assert 'FAST_LINEAR_ATTN_TILELANG_SPEC="${FAST_LINEAR_ATTN_TILELANG_SPEC:-tilelang==0.1.9}"' in wrapper
    assert 'FAST_LINEAR_ATTN_TVM_FFI_SPEC="${FAST_LINEAR_ATTN_TVM_FFI_SPEC:-apache-tvm-ffi==0.1.10}"' in wrapper
    assert 'FAST_LINEAR_ATTN_CUDA_ARCH_LIST="${FAST_LINEAR_ATTN_CUDA_ARCH_LIST:-8.0}"' in wrapper
