import pytest

from starVLA.model.modules.vlm import QWen3_5


def _patch_flash(monkeypatch, *, cuda=True, fa2=False, fa4=False, fa4_gpu_supported=True):
    monkeypatch.setattr(QWen3_5.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(QWen3_5, "is_flash_attn_2_available", lambda: fa2)
    monkeypatch.setattr(QWen3_5, "is_flash_attn_4_available", lambda: fa4)
    monkeypatch.setattr(
        QWen3_5._QWen3_5_Interface,
        "_flash_attn_4_gpu_supported",
        staticmethod(lambda max_head_dim=None: fa4_gpu_supported),
    )
    monkeypatch.setattr(
        QWen3_5._QWen3_5_Interface,
        "_prepare_flash_attn_4",
        staticmethod(lambda max_head_dim=None: fa4 and fa4_gpu_supported),
    )


def test_flash2_request_uses_flash4_when_flash2_unavailable(monkeypatch):
    _patch_flash(monkeypatch, fa2=False, fa4=True)

    backend = QWen3_5._QWen3_5_Interface._resolve_attn_implementation("flash_attention_2")

    assert backend == "flash_attention_4"


def test_explicit_flash4_is_supported(monkeypatch):
    _patch_flash(monkeypatch, fa2=False, fa4=True)

    backend = QWen3_5._QWen3_5_Interface._resolve_attn_implementation("flash_attn_4")

    assert backend == "flash_attention_4"


def test_auto_prefers_flash4(monkeypatch):
    _patch_flash(monkeypatch, fa2=True, fa4=True)

    backend = QWen3_5._QWen3_5_Interface._resolve_attn_implementation("auto")

    assert backend == "flash_attention_4"


@pytest.mark.parametrize("legacy_env_value", [None, "0", "1"])
def test_explicit_sdpa_is_never_promoted_by_ambient_environment(
    monkeypatch,
    legacy_env_value,
):
    _patch_flash(monkeypatch, fa2=True, fa4=True)
    if legacy_env_value is None:
        monkeypatch.delenv("STARVLA_DISABLE_FLASH_ATTN_PROMOTION", raising=False)
    else:
        monkeypatch.setenv(
            "STARVLA_DISABLE_FLASH_ATTN_PROMOTION",
            legacy_env_value,
        )

    backend = QWen3_5._QWen3_5_Interface._resolve_attn_implementation(
        "sdpa",
        strict=True,
    )

    assert backend == "sdpa"


def test_explicit_sdpa_does_not_probe_or_prepare_flash_kernels(monkeypatch):
    _patch_flash(monkeypatch, fa2=True, fa4=True)

    def _unexpected_flash_probe(max_head_dim=None):
        raise AssertionError("an explicit SDPA request must not prepare FlashAttention")

    monkeypatch.setattr(
        QWen3_5._QWen3_5_Interface,
        "_prepare_flash_attn_4",
        staticmethod(_unexpected_flash_probe),
    )

    assert (
        QWen3_5._QWen3_5_Interface._resolve_attn_implementation("sdpa")
        == "sdpa"
    )


def test_strict_flash2_does_not_substitute_flash4(monkeypatch):
    _patch_flash(monkeypatch, fa2=False, fa4=True)

    with pytest.raises(RuntimeError, match="FlashAttention was requested"):
        QWen3_5._QWen3_5_Interface._resolve_attn_implementation(
            "flash_attention_2",
            strict=True,
        )


def test_flash4_install_is_ignored_on_unsupported_gpu(monkeypatch):
    _patch_flash(monkeypatch, fa2=False, fa4=True, fa4_gpu_supported=False)
    monkeypatch.setattr(QWen3_5.torch.cuda, "get_device_capability", lambda: (12, 0))

    backend = QWen3_5._QWen3_5_Interface._resolve_attn_implementation("flash_attention_4")

    assert backend == "sdpa"


def test_flash4_sm120_is_ignored_when_head_dim_is_too_large(monkeypatch):
    _patch_flash(monkeypatch, fa2=False, fa4=True)
    monkeypatch.setattr(
        QWen3_5._QWen3_5_Interface,
        "_flash_attn_4_gpu_supported",
        staticmethod(lambda max_head_dim=None: max_head_dim is None or max_head_dim <= 128),
    )
    monkeypatch.setattr(
        QWen3_5._QWen3_5_Interface,
        "_prepare_flash_attn_4",
        staticmethod(lambda max_head_dim=None: max_head_dim is None or max_head_dim <= 128),
    )
    monkeypatch.setattr(QWen3_5.torch.cuda, "get_device_capability", lambda: (12, 0))

    backend = QWen3_5._QWen3_5_Interface._resolve_attn_implementation(
        "flash_attention_4",
        max_head_dim=256,
    )

    assert backend == "sdpa"
