"""Per-micro-batch padding trim in GRPO log-prob computation (CarbonTeq)."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from trl.trainer.grpo_trainer import GRPOTrainer, _pad_completion, _trim_to_real_tokens


def _model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(config).eval()


def _fake_trainer(model, chunk):
    trainer = SimpleNamespace(
        temperature=1.0,
        model_kwarg_keys={"input_ids", "attention_mask", "logits_to_keep"},
        logits_chunk_size=chunk,
        accelerator=SimpleNamespace(unwrap_model=lambda m: m, is_main_process=True),
        _entropy_bonus_enabled=False,
        _is_vlm=False,
    )
    trainer._get_last_hidden_state = lambda *a, **k: GRPOTrainer._get_last_hidden_state(trainer, *a, **k)
    return trainer


def _batch():
    # Two rows padded to the generation batch's widest prompt (6) and completion (5):
    # row 0 has a 3-token prompt and 2-token completion, row 1 fills both.
    prompt = torch.tensor([[0, 0, 0, 11, 12, 13], [21, 22, 23, 24, 25, 26]])
    prompt_mask = torch.tensor([[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    completion = torch.tensor([[31, 32, 0, 0, 0], [41, 42, 43, 44, 45]])
    completion_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    return torch.cat([prompt, completion], 1), torch.cat([prompt_mask, completion_mask], 1), completion_mask


def test_trim_keeps_only_the_real_extent_of_a_row():
    input_ids, attention_mask, _ = _batch()
    ids, mask, keep = _trim_to_real_tokens(input_ids[:1], attention_mask[:1], 5)
    assert keep == 2
    assert ids.tolist() == [[11, 12, 13, 31, 32]]
    assert mask.tolist() == [[1, 1, 1, 1, 1]]
    assert _pad_completion(torch.ones(1, 2), 5).tolist() == [[1, 1, 0, 0, 0]]


@pytest.mark.parametrize("chunk", [None, 2])
def test_trimmed_logps_match_the_padded_computation_at_real_tokens(chunk):
    model = _model()
    input_ids, attention_mask, completion_mask = _batch()
    trainer = _fake_trainer(model, chunk)
    with torch.no_grad():
        trimmed, entropies, _ = GRPOTrainer._get_per_token_logps_and_entropies(
            trainer, model, input_ids, attention_mask, 5, batch_size=1, compute_entropy=True
        )
        # Reference: each row alone at its real extent, positions starting at 0 as
        # generation starts them.
        for row, (prompt_len, completion_len) in enumerate([(3, 2), (6, 5)]):
            real = input_ids[row][attention_mask[row].bool()].unsqueeze(0)
            logits = model(real).logits[:, prompt_len - 1 : -1, :]
            expected = torch.log_softmax(logits.float(), -1).gather(-1, real[:, prompt_len:, None]).squeeze(-1)
            torch.testing.assert_close(trimmed[row, :completion_len], expected[0], rtol=1e-4, atol=1e-4)
    assert trimmed.shape == entropies.shape == (2, 5)
    assert torch.all(trimmed[0, 2:] == 0)
