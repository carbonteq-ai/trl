import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    _load_logitsprocs_by_fqcns,
)

from trl.generation.constrained_replay import (
    ConstrainedReplayLogitsProcessor,
    collect_constrained_replay_results,
)


def test_constrained_replay_processor_uses_vllm_fqcn_contract():
    loaded = _load_logitsprocs_by_fqcns(
        ["trl.generation.constrained_replay:ConstrainedReplayLogitsProcessor"]
    )

    assert loaded == [ConstrainedReplayLogitsProcessor]


class _Matcher:
    def __init__(self, _context):
        self.position = 0

    def fill_next_token_bitmask(self, bitmask, row):
        bitmask[row].fill_(False)
        allowed = ([1, 2], [3])[self.position]
        bitmask[row, allowed] = True

    def accept_token(self, token_id):
        allowed = ([1, 2], [3])[self.position]
        if token_id not in allowed:
            return False
        self.position += 1
        return True


class _XGrammar:
    GrammarMatcher = _Matcher

    @staticmethod
    def allocate_token_bitmask(rows, vocab_size):
        return torch.zeros((rows, vocab_size), dtype=torch.bool)

    @staticmethod
    def apply_token_bitmask_inplace(logits, bitmask):
        logits.masked_fill_(~bitmask, float("-inf"))


class _Compiler:
    @staticmethod
    def compile_json_schema(schema, any_whitespace):
        assert json.loads(schema) == {"type": "array"}
        assert any_whitespace is False
        return object()


def test_exact_completion_is_scored_before_forcing_and_evidence_is_collected():
    processor = ConstrainedReplayLogitsProcessor.__new__(ConstrainedReplayLogitsProcessor)
    AdapterLogitsProcessor.__init__(processor, SimpleNamespace(), torch.device("cpu"), False)
    processor._compiler = _Compiler()
    processor._xgr = _XGrammar()
    processor._device = torch.device("cpu")
    processor._vocab_size = 4
    schema = {"type": "array"}
    schema_digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    params = SamplingParams(
        extra_args={
            "constrained_replay": {
                "request_id": "row-1",
                "completion_ids": [2, 3],
                "json_schema": schema,
                "schema_digest": schema_digest,
            }
        }
    )
    replay = processor.new_req_logits_processor(params)
    assert replay is not None

    first_logits = torch.tensor([0.0, 1.0, 2.0, 4.0])
    replay([], first_logits)
    assert first_logits.tolist() == [float("-inf"), float("-inf"), 0.0, float("-inf")]
    replay([2], torch.tensor([3.0, 2.0, 1.0, 0.0]))

    result = collect_constrained_replay_results(["row-1"])[0]
    assert result["completion_ids"] == [2, 3]
    assert result["allowed_counts"] == [2, 1]
    assert result["logprobs"][0] == pytest.approx(2.0 - torch.logsumexp(torch.tensor([1.0, 2.0]), 0).item())
    assert result["logprobs"][1] == pytest.approx(0.0)
    assert len(result["allowed_set_digests"]) == 2


def test_constrained_replay_rejects_schema_digest_mismatch():
    processor = ConstrainedReplayLogitsProcessor.__new__(ConstrainedReplayLogitsProcessor)
    AdapterLogitsProcessor.__init__(processor, SimpleNamespace(), torch.device("cpu"), False)
    processor._compiler = _Compiler()
    processor._xgr = _XGrammar()
    processor._device = torch.device("cpu")
    processor._vocab_size = 4
    params = SamplingParams(
        extra_args={
            "constrained_replay": {
                "request_id": "row-bad",
                "completion_ids": [2],
                "json_schema": {"type": "array"},
                "schema_digest": "0" * 64,
            }
        }
    )

    with pytest.raises(ValueError, match="schema digest mismatch"):
        processor.new_req_logits_processor(params)
