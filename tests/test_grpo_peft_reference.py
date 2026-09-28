"""The KL reference of a trainer that receives an already trained PEFT adapter."""

import copy

import pytest
import torch
from datasets import Dataset
from transformers.utils import is_peft_available

from trl import GRPOConfig, GRPOTrainer, RLOOConfig


if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model

pytestmark = pytest.mark.skipif(not is_peft_available(), reason="PEFT is required")


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<pad>": 0, "<unk>": 1, "prompt": 2, "answer": 3, "<eos>": 4}, unk_token="<unk>")
        ),
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
    )


def _trained_adapter_model(tmp_path):
    """A base model and the same base carrying a non-zero adapter loaded as `--model-from-run` loads it."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    base = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=5,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    reference_base = copy.deepcopy(base)
    # init_lora_weights=False makes the adapter non-zero, like a checkpoint from an earlier run.
    trained = get_peft_model(
        copy.deepcopy(base), LoraConfig(r=2, target_modules=["c_attn"], init_lora_weights=False, fan_in_fan_out=True)
    )
    trained.save_pretrained(tmp_path / "adapter")
    policy = PeftModel.from_pretrained(copy.deepcopy(base), tmp_path / "adapter", is_trainable=True)
    return reference_base, policy


def _trainer(tmp_path, policy, **config):
    def rollout(prompts, trainer, inputs):
        return {
            "prompt_ids": [[2]] * len(inputs),
            "completion_ids": [[3, 4]] * len(inputs),
            "logprobs": None,
        }

    return GRPOTrainer(
        model=policy,
        reward_funcs=lambda completions, **kwargs: [float(index % 2) for index in range(len(completions))],
        args=GRPOConfig(
            output_dir=str(tmp_path / "out"),
            use_cpu=True,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=2,
            temperature=1.0,
            beta=0.1,
            report_to="none",
            save_strategy="no",
            gradient_checkpointing=False,
            **config,
        ),
        train_dataset=Dataset.from_dict({"prompt": ["prompt"] * 2}),
        processing_class=_tokenizer(),
        rollout_func=rollout,
    )


def _completion_logps(model, trainer_output):
    input_ids = torch.cat([trainer_output["prompt_ids"], trainer_output["completion_ids"]], dim=1)
    attention_mask = torch.cat([trainer_output["prompt_mask"], trainer_output["completion_mask"]], dim=1)
    width = trainer_output["completion_ids"].shape[1]
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -width - 1 : -1]
    return logits.log_softmax(-1).gather(-1, trainer_output["completion_ids"].unsqueeze(-1)).squeeze(-1)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_base_reference_uses_the_base_model_for_a_trained_adapter(tmp_path):
    reference_base, policy = _trained_adapter_model(tmp_path)
    trainer = _trainer(tmp_path, policy, peft_reference="base")

    assert "ref" not in trainer.model.peft_config
    output = trainer._generate_and_score_completions([{"prompt": "prompt"}, {"prompt": "prompt"}])

    mask = output["completion_mask"].bool()
    expected = _completion_logps(reference_base, output)
    torch.testing.assert_close(output["ref_per_token_logps"][mask], expected[mask])
    # The trained adapter moves the policy, so the reference is not the starting checkpoint.
    adapter = _completion_logps(trainer.model, output)
    assert not torch.allclose(adapter[mask], expected[mask])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_default_reference_is_a_frozen_copy_of_the_starting_adapter(tmp_path):
    reference_base, policy = _trained_adapter_model(tmp_path)
    trainer = _trainer(tmp_path, policy)

    assert trainer.args.peft_reference == "adapter_copy"
    assert "ref" in trainer.model.peft_config
    output = trainer._generate_and_score_completions([{"prompt": "prompt"}, {"prompt": "prompt"}])

    mask = output["completion_mask"].bool()
    starting_adapter = _completion_logps(trainer.model, output)
    torch.testing.assert_close(output["ref_per_token_logps"][mask], starting_adapter[mask])
    assert not torch.allclose(output["ref_per_token_logps"][mask], _completion_logps(reference_base, output)[mask])


@pytest.mark.parametrize("config_type", [GRPOConfig, RLOOConfig])
def test_peft_reference_is_validated(config_type):
    assert config_type(output_dir="/tmp/trl-test", report_to="none", peft_reference="base").peft_reference == "base"
    with pytest.raises(ValueError, match="peft_reference must be 'adapter_copy' or 'base'"):
        config_type(output_dir="/tmp/trl-test", report_to="none", peft_reference="start")
