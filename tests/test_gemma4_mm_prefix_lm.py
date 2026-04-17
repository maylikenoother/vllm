from types import SimpleNamespace

from vllm.model_executor.models.config import Gemma4Config


def test_gemma4_sets_mm_prefix_lm_for_vision_bidirectional_attention():
    hf_text_config = SimpleNamespace(use_bidirectional_attention="vision")
    hf_config = SimpleNamespace(model_type="gemma4")
    model_config = SimpleNamespace(hf_text_config=hf_text_config, hf_config=hf_config)
    attention_config = SimpleNamespace(backend=None)
    vllm_config = SimpleNamespace(model_config=model_config, attention_config=attention_config)

    Gemma4Config.verify_and_update_config(vllm_config)

    assert getattr(hf_config, "is_mm_prefix_lm", False) is True


def test_gemma4_does_not_set_mm_prefix_lm_when_not_vision_bidirectional_attention():
    hf_text_config = SimpleNamespace(use_bidirectional_attention=False)
    hf_config = SimpleNamespace(model_type="gemma4")
    model_config = SimpleNamespace(hf_text_config=hf_text_config, hf_config=hf_config)
    attention_config = SimpleNamespace(backend=None)
    vllm_config = SimpleNamespace(model_config=model_config, attention_config=attention_config)

    Gemma4Config.verify_and_update_config(vllm_config)

    assert hasattr(hf_config, "is_mm_prefix_lm") is False
