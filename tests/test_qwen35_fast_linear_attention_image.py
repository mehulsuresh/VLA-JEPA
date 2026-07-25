import argparse
import importlib.util
import json
import os
import subprocess
import sys
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


def test_default_transformers_binding_validation_still_fails_unbound():
    probe = _load_probe_module()
    package_bindings = {
        name: (lambda: None) for name in probe.REQUIRED_MODELING_BINDINGS
    }
    unbound_modeling = SimpleNamespace(
        **{name: None for name in probe.REQUIRED_MODELING_BINDINGS}
    )

    with pytest.raises(RuntimeError, match="did not bind every Qwen3.5 fast-path"):
        probe._resolve_transformers_bindings(unbound_modeling, package_bindings)


def test_cpu_build_mode_permits_only_unbound_transformers_globals_without_cuda(
    monkeypatch,
):
    probe = _load_probe_module()
    package_bindings = {
        name: (lambda: None) for name in probe.REQUIRED_MODELING_BINDINGS
    }
    unbound_modeling = SimpleNamespace(
        **{name: None for name in probe.REQUIRED_MODELING_BINDINGS}
    )
    monkeypatch.setattr(probe, "_cuda_is_available", lambda: False)

    bindings, validation = probe._resolve_transformers_bindings(
        unbound_modeling,
        package_bindings,
        allow_unbound_for_cpu_build=True,
    )

    assert bindings == package_bindings
    assert validation == {
        "mode": "cpu-build-no-cuda",
        "bound": [],
        "unbound": list(probe.REQUIRED_MODELING_BINDINGS),
    }


def test_cpu_build_exception_is_rejected_when_cuda_is_available(monkeypatch):
    probe = _load_probe_module()
    package_bindings = {
        name: (lambda: None) for name in probe.REQUIRED_MODELING_BINDINGS
    }
    unbound_modeling = SimpleNamespace(
        **{name: None for name in probe.REQUIRED_MODELING_BINDINGS}
    )
    monkeypatch.setattr(probe, "_cuda_is_available", lambda: True)

    with pytest.raises(RuntimeError, match="valid only when CUDA is unavailable"):
        probe._resolve_transformers_bindings(
            unbound_modeling,
            package_bindings,
            allow_unbound_for_cpu_build=True,
        )


def test_cpu_build_mode_rejects_wrong_transformers_implementation(monkeypatch):
    probe = _load_probe_module()
    package_bindings = {
        name: (lambda: None) for name in probe.REQUIRED_MODELING_BINDINGS
    }
    modeling_values = {name: None for name in probe.REQUIRED_MODELING_BINDINGS}
    modeling_values["chunk_gated_delta_rule"] = lambda: None
    monkeypatch.setattr(probe, "_cuda_is_available", lambda: False)

    with pytest.raises(RuntimeError, match="unexpected implementations.*chunk_gated"):
        probe._resolve_transformers_bindings(
            SimpleNamespace(**modeling_values),
            package_bindings,
            allow_unbound_for_cpu_build=True,
        )


def test_cpu_build_mode_rejects_missing_transformers_global(monkeypatch):
    probe = _load_probe_module()
    package_bindings = {
        name: (lambda: None) for name in probe.REQUIRED_MODELING_BINDINGS
    }
    modeling_values = {name: None for name in probe.REQUIRED_MODELING_BINDINGS}
    del modeling_values["fused_recurrent_gated_delta_rule"]
    monkeypatch.setattr(probe, "_cuda_is_available", lambda: False)

    with pytest.raises(
        RuntimeError,
        match="missing required CPU-build globals.*fused_recurrent",
    ):
        probe._resolve_transformers_bindings(
            SimpleNamespace(**modeling_values),
            package_bindings,
            allow_unbound_for_cpu_build=True,
        )


def test_cpu_build_main_flag_skips_kernel_probe(monkeypatch, capsys):
    probe = _load_probe_module()
    received = {}

    def fake_load_integrations(*_args, **kwargs):
        received.update(kwargs)
        return (
            {"flash-linear-attention": "0.5.1"},
            SimpleNamespace(),
            type("FakeQwenConfig", (), {}),
            {},
            {"mode": "cpu-build-no-cuda", "bound": [], "unbound": []},
            {},
        )

    monkeypatch.setattr(probe, "_load_integrations", fake_load_integrations)
    monkeypatch.setattr(
        probe,
        "_run_kernel_probe",
        lambda *_args, **_kwargs: pytest.fail("CPU build mode ran CUDA kernels"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PROBE_PATH), "--cpu-build-imports-only"],
    )

    assert probe.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert received["allow_unbound_transformers_for_cpu_build"] is True
    assert result["mode"] == "cpu-build-imports-only"
    assert "kernel_probe" not in result


def test_probe_fails_closed_on_transformers_version_mismatch(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda _distribution: "5.14.0")

    with pytest.raises(RuntimeError, match="expected '5.13.1', found '5.14.0'"):
        probe._require_exact_package_version("transformers", "5.13.1")


def test_imports_only_loads_tilelang_tvm_ffi_and_qwen35_bindings(monkeypatch):
    probe = _load_probe_module()
    binding_values = {
        name: (type(f"Binding_{name}", (), {}) if name == "FusedRMSNormGated" else lambda: None)
        for name in probe.REQUIRED_MODELING_BINDINGS
    }

    class FakeQwenConfig:
        pass

    modules = {
        "tilelang": SimpleNamespace(__name__="tilelang", __file__="/runtime/tilelang.py"),
        "tvm_ffi": SimpleNamespace(__name__="tvm_ffi", __file__="/runtime/tvm_ffi.py"),
        "causal_conv1d": SimpleNamespace(
            __name__="causal_conv1d",
            __file__="/runtime/causal_conv1d.py",
            causal_conv1d_fn=binding_values["causal_conv1d_fn"],
            causal_conv1d_update=binding_values["causal_conv1d_update"],
        ),
        "fla.modules": SimpleNamespace(
            __name__="fla.modules",
            __file__="/runtime/fla/modules.py",
            FusedRMSNormGated=binding_values["FusedRMSNormGated"],
        ),
        "fla.ops.gated_delta_rule": SimpleNamespace(
            __name__="fla.ops.gated_delta_rule",
            __file__="/runtime/fla/ops/gated_delta_rule.py",
            chunk_gated_delta_rule=binding_values["chunk_gated_delta_rule"],
            fused_recurrent_gated_delta_rule=binding_values[
                "fused_recurrent_gated_delta_rule"
            ],
        ),
        "transformers.models.qwen3_5.modeling_qwen3_5": SimpleNamespace(
            __name__="transformers.models.qwen3_5.modeling_qwen3_5",
            __file__="/runtime/transformers/modeling_qwen3_5.py",
            **binding_values,
        ),
        "transformers.models.qwen3_5.configuration_qwen3_5": SimpleNamespace(
            __name__="transformers.models.qwen3_5.configuration_qwen3_5",
            __file__="/runtime/transformers/configuration_qwen3_5.py",
            Qwen3_5TextConfig=FakeQwenConfig,
        ),
    }
    expected_versions = {
        "flash-linear-attention": "0.5.1",
        "causal-conv1d": "1.6.2.post1",
        "transformers": "5.13.1",
        "tilelang": "0.1.9",
        "apache-tvm-ffi": "0.1.10",
    }
    imported = []

    monkeypatch.setattr(
        probe.importlib.metadata,
        "version",
        lambda distribution: expected_versions[distribution],
    )

    def fake_import(module_name):
        imported.append(module_name)
        return modules[module_name]

    monkeypatch.setattr(probe.importlib, "import_module", fake_import)

    (
        versions,
        modeling,
        config_cls,
        bindings,
        binding_validation,
        import_report,
    ) = probe._load_integrations(
        "0.5.1", "1.6.2.post1", "5.13.1", "0.1.9", "0.1.10"
    )

    assert versions == expected_versions
    assert config_cls is FakeQwenConfig
    assert modeling is modules["transformers.models.qwen3_5.modeling_qwen3_5"]
    assert bindings == binding_values
    assert binding_validation == {
        "mode": "strict-transformers-identity",
        "bound": list(probe.REQUIRED_MODELING_BINDINGS),
        "unbound": [],
    }
    assert "tilelang" in imported
    assert "tvm_ffi" in imported
    assert "transformers.models.qwen3_5.modeling_qwen3_5" in imported
    assert "transformers.models.qwen3_5.configuration_qwen3_5" in imported
    assert import_report["tilelang"]["module"] == "tilelang"
    assert import_report["apache-tvm-ffi"]["module"] == "tvm_ffi"


def test_runtime_import_failure_names_distribution_and_module(monkeypatch):
    probe = _load_probe_module()

    def fail_import(_module_name):
        raise ImportError("binary ABI mismatch")

    monkeypatch.setattr(probe.importlib, "import_module", fail_import)
    with pytest.raises(
        RuntimeError,
        match="apache-tvm-ffi.*'tvm_ffi'.*ImportError: binary ABI mismatch",
    ):
        probe._import_runtime_module("apache-tvm-ffi", "tvm_ffi")


def test_pip_check_allows_only_exact_known_moge_optional_failures():
    probe = _load_probe_module()
    assert probe.KNOWN_MOGE_OPTIONAL_DEPENDENCY_FAILURES == frozenset(
        {
            "moge 2.0.0 requires gradio, which is not installed.",
            "moge 2.0.0 requires opencv-python, which is not installed.",
        }
    )
    known = "\n".join(sorted(probe.KNOWN_MOGE_OPTIONAL_DEPENDENCY_FAILURES))

    result = probe._validate_pip_check_result(
        1,
        known,
        "",
        allow_known_moge_optional_dependencies=True,
    )

    assert result["status"] == "ok"
    assert result["pip_check_returncode"] == 1
    assert result["allowed_failures"] == sorted(
        probe.KNOWN_MOGE_OPTIONAL_DEPENDENCY_FAILURES
    )


@pytest.mark.parametrize(
    ("returncode", "stdout", "allow", "message"),
    [
        (
            1,
            "moge 2.0.0 requires gradio, which is not installed.\n"
            "unrelated 1.0 requires missing-package, which is not installed.",
            True,
            "unapproved dependency failures",
        ),
        (
            1,
            "moge 2.0.0 requires gradio, which is not installed.",
            False,
            "unapproved dependency failures",
        ),
        (
            0,
            "moge 2.0.0 requires gradio, which is not installed.",
            True,
            "returned success while reporting dependency failures",
        ),
        (2, "", True, "without an approved diagnostic"),
    ],
)
def test_pip_check_policy_fails_closed(returncode, stdout, allow, message):
    probe = _load_probe_module()

    with pytest.raises(RuntimeError, match=message):
        probe._validate_pip_check_result(
            returncode,
            stdout,
            "",
            allow_known_moge_optional_dependencies=allow,
        )


def test_hopper_reference_metric_gate_rejects_wrong_gradients():
    probe = _load_probe_module()
    good = {"relative_l2": 0.03, "cosine": 0.999, "max_abs": 0.01, "reference_norm": 1.0}
    probe._validate_reference_metrics(
        "q_gradient", good, max_relative_l2=0.2, min_cosine=0.98
    )

    with pytest.raises(RuntimeError, match="q_gradient.*relative_l2"):
        probe._validate_reference_metrics(
            "q_gradient",
            {**good, "relative_l2": 0.21},
            max_relative_l2=0.2,
            min_cosine=0.98,
        )
    with pytest.raises(RuntimeError, match="q_gradient.*cosine"):
        probe._validate_reference_metrics(
            "q_gradient",
            {**good, "cosine": 0.97},
            max_relative_l2=0.2,
            min_cosine=0.98,
        )


def test_hopper_reference_crosses_chunk_boundary_and_is_in_kernel_probe():
    probe = _load_probe_module()
    source = PROBE_PATH.read_text(encoding="utf-8")

    assert probe.HOPPER_REFERENCE_SEQUENCE_LENGTH == 65
    assert "_pytorch_gated_delta_rule_reference" in source
    assert "_probe_hopper_chunk_gradient_reference" in source
    assert 'bindings["chunk_gated_delta_rule"]' in source
    assert '"hopper_chunk_gradient_reference"' in source


def test_probe_help_does_not_require_ml_runtime_imports():
    result = subprocess.run(
        ["python", str(PROBE_PATH), "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--imports-only" in result.stdout
    assert "--cpu-build-imports-only" in result.stdout
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
    assert 'SHELL ["/bin/bash", "-Eeuo", "pipefail", "-c"]' in dockerfile
    assert "probe_qwen35_fast_linear_attention.py" in dockerfile
    assert "--cpu-build-imports-only" in dockerfile
    assert "--imports-only" not in dockerfile
    assert '"${FAST_LINEAR_ATTN_TVM_FFI_SPEC}" "${FAST_LINEAR_ATTN_TILELANG_SPEC}"' in dockerfile
    assert "python -m pip check" not in dockerfile
    assert "--check-python-dependencies" in dockerfile
    assert "--allow-known-moge-optional-deps" in dockerfile
    assert dockerfile.index("python -m pip install -e .") < dockerfile.index(
        "--check-python-dependencies"
    )
    assert '--expected-transformers-version "${FAST_LINEAR_ATTN_TRANSFORMERS_SPEC##*==}"' in dockerfile
    assert 'INSTALL_FAST_LINEAR_ATTN="${INSTALL_FAST_LINEAR_ATTN:-0}"' in wrapper
    assert 'FAST_LINEAR_ATTN_TRANSFORMERS_SPEC="${FAST_LINEAR_ATTN_TRANSFORMERS_SPEC:-transformers==5.13.1}"' in wrapper
    assert 'FAST_LINEAR_ATTN_TILELANG_SPEC="${FAST_LINEAR_ATTN_TILELANG_SPEC:-tilelang==0.1.9}"' in wrapper
    assert 'FAST_LINEAR_ATTN_TVM_FFI_SPEC="${FAST_LINEAR_ATTN_TVM_FFI_SPEC:-apache-tvm-ffi==0.1.10}"' in wrapper
    assert 'FAST_LINEAR_ATTN_CUDA_ARCH_LIST="${FAST_LINEAR_ATTN_CUDA_ARCH_LIST:-8.0}"' in wrapper
