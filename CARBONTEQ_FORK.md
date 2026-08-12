# CarbonTeq TRL fork ledger

This ledger records the maintained, generally reusable delta between
`carbonteq-ai/trl` and `huggingface/trl`. Model selections, environment policy,
job configuration, and qualification evidence remain in the consuming
Posttrain framework.

Fork status: `candidate`. `trl==1.9.2.post8` is the current published CarbonTeq
build, tagged at commit `9a219ce5a593d85fe6058025de211ce42267e6b6` as
`carbonteq-v1.9.2.post8`. Its exact wheel SHA-256 is
`9933755547f3d09e5abef195607e5d8b682d277a4fc1febce6bd2cd053140ca5` and its
source-distribution SHA-256 is
`c6e167ffafae776e5c9514e9a8777058c0c05835afe568caf4e063df962bb06c`.

## Upstream base

- Upstream repository: `https://github.com/huggingface/trl`
- Upstream base: `33f9e462728b98f7f91d38b99328e81adde2faa0` (`v1.9.2`)
- CarbonTeq repository: `https://github.com/carbonteq-ai/trl`
- Release branch: `v1.9.2.post5-release` (publishing `1.9.2.post9`; the
  already-reserved `carbonteq-v1.9.2.post5` through `post7` tags name
  unrelated commits)
- Published package release: `trl==1.9.2.post8`
- Published release commit: `9a219ce5a593d85fe6058025de211ce42267e6b6`

The Posttrain dependency declaration and lockfile are the executable consumer
authority. A candidate capability is not published until its fork commit,
immutable release tag, package hashes, and clean-install verification exist.

## Maintained delta

The fork keeps the following behavior on top of upstream 1.9.2:

- vLLM request waves and explicit resident sequence/token caps so a large
  logical rollout batch does not become an unbounded scheduler burst;
- native MTP and guarded vLLM engine configuration, with per-generation
  speculative-decoding counters accumulated into trainer metrics;
- TurboQuant KV-cache compatibility at the vLLM boundary without changing
  trainable actor weights;
- composite model weight-name prefixes and native LoRA-only synchronization,
  including compatible sleep/wake behavior for colocated vLLM;
- a mandatory first-rollout actor/sampler log-probability parity gate, with a
  globally token-weighted tolerance;
- separate raw actor/sampler parity and processed behavior-policy evidence:
  vLLM teacher-forces a bounded prompt/completion probe for the synchronization
  gate, while sampled post-processor log probabilities remain available to
  token-level TIS;
- bounded DAPO dynamic sampling, configurable reward scaling, correct exclusion
  of truncated completions from group statistics, and advantage diagnostics;
- bounded active sampling that requests only the synchronized number of missing
  prompt groups, plus a first-class `Olmo3GRPOConfig` for the published
  model-agnostic OlmoRL recipe with immutable objective-defining settings;
- optional token-aligned precomputed advantages for agentic estimators without
  moving environment ownership into TRL;
- exact-token external rollout hooks and source-row preservation for
  `IWOPDTrainer`;
- IW-OPD support for sparse environment masks, cached rollout log-probabilities,
  LoRA-only vLLM synchronization, speculative configuration, and engine kwargs;
- complete behavior-policy sampling for IW-OPD: top-p, top-k, min-p,
  repetition penalty, and additional transformers/vLLM generation arguments
  such as presence penalty are represented by `IWOPDConfig`, retained by
  `IWOPDTrainer`, and forwarded to both generation engines;
- IW-OPD validates required student, teacher, and externally supplied
  behavior-policy log probabilities before constructing an importance weight,
  so an invalid rollout or actor state fails at the owning numerical boundary
  with the affected-token count rather than surfacing later as an opaque
  aggregate non-finite loss;
- IW-OPD validates the token loss after combining those finite inputs and
  reports the student, teacher, behavior-policy, advantage, and weight ranges
  when float32 reduction overflows;
- memory-bounded log-probability projection for GRPO policy/reference scoring;
- compatibility with the Posttrain runtime's `datasets>=4.6.1,<4.7` constraint.

Upstream 1.9.2 already contains the corrected DAPO loss normalizer,
truncated-completion normalization, cross-rank metric aggregation, and the
IW-OPD trainer split. Those behaviors are not duplicated as fork patches.

## Ownership and semantics

`GRPOTrainer` owns GRPO-family policy updates, including the fork's DAPO
controls. `IWOPDTrainer` owns Posttrain's verifier-driven on-policy
distillation. The stable upstream `DistillationTrainer` remains the
full-vocabulary distillation API and is not extended with environment rollout
semantics.

The external IW-OPD rollout callback returns exact prompt/completion token IDs,
sampling log-probabilities, a completion loss mask, and optional unique rollout
IDs. Tool and environment tokens may remain in the completion sequence for
causal context while the mask excludes them from the update. Returned sampling
log-probabilities are stored at full sequence width so IW-OPD advantages remain
aligned after prompt slicing.

Native MTP accelerates rollout generation through a model-provided draft head;
it does not add an MTP auxiliary training loss. TurboQuant applies only to the
rollout KV cache. LoRA-only synchronization requires colocated vLLM and a PEFT
student; the base model remains immutable while the active adapter is refreshed.

DAPO dynamic sampling remains opt-in, bounded, and group-preserving. Truncated
completions may be masked from the update and are excluded from the group mean
and standard deviation when configured. Scalar verifier components remain a
single reward before DAPO normalization; component metrics are diagnostic, not
independent objectives.

The actor/sampler gate compares like with like. Sampling temperature, top-p,
top-k, repetition, and presence processors intentionally alter behavior-policy
log probabilities, so those values are not compared directly with raw actor
logits. `VLLMGeneration.score_completion_logprobs` teacher-forces a
deterministic, token-bounded probe through vLLM prompt-logprob collection;
`GRPOTrainer` recomputes the same raw actor values at temperature 1. The
existing processed delta remains the TIS input and diagnostic. Truncated or
masked rows are excluded from the parity probe.

## Compatibility constraints

- Package baseline: Python 3.10+, Transformers as declared by upstream 1.9.2,
  and vLLM `>=0.17.0,<=0.25.1`.
- Posttrain qualification target: Python 3.13, Torch 2.11, Transformers 5.14,
  vLLM 0.25.1, PEFT 0.19, and `datasets>=4.6.1,<4.7`.
- External vLLM server mode must receive equivalent engine settings at server
  launch; colocated-only settings are rejected where equivalence is impossible.
- `vllm_weight_sync_mode="lora"` is colocated-only.
- `distillation_objective="iw_opd"` is fully on-policy and requires fresh
  rollout log-probabilities aligned to the exact sampled tokens.
- `IWOPDConfig.generation_kwargs` may override its declared generation controls
  on both local transformers and vLLM paths. A consumer that selects a control
  must use a compatible engine; unsupported engine parameters fail at the
  engine boundary rather than being silently discarded.
- TurboQuant and MTP are configured and qualified independently before their
  combination is promoted.

## Regression tests

Run the focused maintained-delta suite from this repository:

    uv run pytest -q \
      tests/test_vllm_generation.py \
      tests/test_dapo_dynamic_sampling.py \
      tests/test_sampo_precomputed_advantages.py \
      tests/experimental/test_iw_opd_trainer.py

Run the GRPO selections that cover synchronization, parity, truncation,
advantages, and projection:

    uv run pytest -q tests/test_grpo_trainer.py \
      -k 'policy_parity or importance_sampling or truncated or advantage or logits_chunking'

The raw-parity candidate specifically changes
`trl/generation/vllm_generation.py`, `trl/trainer/grpo_config.py`, and
`trl/trainer/grpo_trainer.py`; its regression coverage is in
`tests/test_vllm_generation.py`, `tests/test_vllm_client_server.py`, and
`tests/test_grpo_trainer.py`. Run:

    uv run pytest -q \
      tests/test_grpo_trainer.py \
      tests/test_vllm_generation.py \
      tests/test_vllm_client_server.py \
      -k 'policy_parity or extract_actual_prompt_logprobs or score_completion_logprobs'

Before publication, the exact candidate wheel must also pass a real colocated
LoRA canary that records finite raw parity below the configured tolerance,
independent processed TIS evidence, one optimizer update, and adapter-only
artifacts.

Then run Ruff, the complete TRL test suite, and the Posttrain adapter contract
tests. Publication is manual from Posttrain's repository-scoped retained-asset
workflow: it downloads the immutable GitHub Release assets, verifies their
supplied SHA-256 values, uploads the exact bytes to the internal stable index,
and retains a clean-install receipt. Forks do not receive release runners or
execute fork-controlled publication workflows.

## Rebase procedure

1. Fetch `upstream` and create a fresh `codex/` branch from the intended stable
   tag, never from an unreleased moving branch.
2. Inventory each maintained behavior against upstream and drop any patch whose
   semantics and regression coverage are now upstream.
3. Reapply shared vLLM lifecycle changes before trainer-specific hooks.
4. Port on-policy distillation behavior to the current upstream trainer owner;
   do not restore removed APIs merely to avoid adapting the consumer.
5. Run focused tests after each behavior group, then the full suite and
   Posttrain contract tests.
6. Update this base, constraints, release tag, and consumer documentation before
   publication.

## Publication checklist

Before updating Posttrain:

1. finish the clean fork commit and push it;
2. build the candidate TRL version from the immutable release commit;
3. record wheel and source hashes;
4. create and verify the immutable CarbonTeq tag and release;
5. use Posttrain's manual retained-asset workflow to upload the exact artifacts
   to the internal stable index;
6. verify a clean install imports the expected version and IW-OPD API;
7. update the Posttrain dependency, lockfile, fork consumer page, and release
   receipt;
8. run Posttrain CPU contracts and the selected GPU canaries before promoting a
   Posttrain release.
