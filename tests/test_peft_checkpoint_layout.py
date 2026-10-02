"""Native checkpoint restoration for PEFT's root/subdirectory layout."""

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import GPT2Config, GPT2LMHeadModel

from trl.trainer.base_trainer import _BaseTrainer


def test_resume_restores_root_policy_and_subdirectory_reference(tmp_path):
    torch.manual_seed(19)
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM")
    model = get_peft_model(GPT2LMHeadModel(GPT2Config(
        n_layer=1, n_head=1, n_embd=8, n_positions=16, vocab_size=16,
    )), config)
    model.add_adapter("ref", config)
    model.set_adapter("default")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".default." in name:
                parameter.fill_(0.25)
            elif ".ref." in name:
                parameter.fill_(-0.125)
    expected = {role: {key: value.clone() for key, value in
                      get_peft_model_state_dict(model, adapter_name=role).items()}
                for role in ("default", "ref")}
    model.save_pretrained(tmp_path)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".default." in name or ".ref." in name:
                parameter.zero_()
    trainer = object.__new__(_BaseTrainer)
    trainer.model = model
    trainer.is_fsdp_enabled = False
    trainer.is_deepspeed_enabled = False
    trainer._load_from_checkpoint(str(tmp_path))
    for role in expected:
        restored = get_peft_model_state_dict(model, adapter_name=role)
        assert restored.keys() == expected[role].keys()
        assert all(torch.equal(restored[key], expected[role][key]) for key in restored)
    assert model.active_adapters == ["default"]
    assert all(parameter.requires_grad == (".default." in name)
               for name, parameter in model.named_parameters()
               if ".default." in name or ".ref." in name)
