import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_gemma4_config_cls():
    repo_root = Path(__file__).resolve().parents[1]
    config_py = repo_root / "vllm" / "model_executor" / "models" / "config.py"
    spec = importlib.util.spec_from_file_location(
        "vllm_model_executor_models_config",
        config_py,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Gemma4Config


def test_gemma4_sets_mm_prefix_lm_for_vision_bidirectional_attention():
    Gemma4Config = _load_gemma4_config_cls()
    hf_text_config = SimpleNamespace(use_bidirectional_attention="vision")
    hf_config = SimpleNamespace(model_type="gemma4")
    model_config = SimpleNamespace(hf_text_config=hf_text_config, hf_config=hf_config)
    attention_config = SimpleNamespace(backend=None)
    vllm_config = SimpleNamespace(model_config=model_config, attention_config=attention_config)

    Gemma4Config.verify_and_update_config(vllm_config)

    assert getattr(model_config, "is_mm_prefix_lm", False) is True
    assert getattr(hf_config, "is_mm_prefix_lm", False) is True


def test_gemma4_does_not_set_mm_prefix_lm_when_not_vision_bidirectional_attention():
    Gemma4Config = _load_gemma4_config_cls()
    hf_text_config = SimpleNamespace(use_bidirectional_attention=False)
    hf_config = SimpleNamespace(model_type="gemma4")
    model_config = SimpleNamespace(hf_text_config=hf_text_config, hf_config=hf_config)
    attention_config = SimpleNamespace(backend=None)
    vllm_config = SimpleNamespace(model_config=model_config, attention_config=attention_config)

    Gemma4Config.verify_and_update_config(vllm_config)

    assert hasattr(model_config, "is_mm_prefix_lm") is False
    assert hasattr(hf_config, "is_mm_prefix_lm") is False
