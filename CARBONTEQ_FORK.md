# CarbonTeq TRL fork ledger

This ledger records the maintained, generally reusable delta between
`carbonteq-ai/trl` and `huggingface/trl`. Model selections, environment policy,
job configuration, and qualification evidence remain in the consuming
Posttrain framework.

## Upstream base

- Upstream repository: `https://github.com/huggingface/trl`
- Upstream base: `33f9e462728b98f7f91d38b99328e81adde2faa0` (`v1.9.2`)
- CarbonTeq repository: `https://github.com/carbonteq-ai/trl`
- Development branch: `feat/iwopd-native-template-constrained-logprobs`
- Intended package release: `trl==1.9.2.post5`
- Release implementation commit: `2371bca979a0d067c88c7a46ad76449ea458fc00`

The Posttrain dependency declaration and lockfile are the executable consumer
authority. Do not update them or describe a candidate capability as published
until the fork commit, immutable release tag, package hash, and clean-install
verification exist.

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
- bounded DAPO dynamic sampling, configurable reward scaling, correct exclusion
  of truncated completions from group statistics, and advantage diagnostics;
- bounded active sampling that requests only the synchronized number of missing
  prompt groups, plus a first-class `Olmo3GRPOConfig` for the published
  model-agnostic OlmoRL recipe with immutable objective-defining settings;
- optional token-aligned precomputed advantages for agentic estimators without
  moving environment ownership into TRL;
- exact-token external rollout hooks and source-row preservation for
  `IWOPDTrainer`;
- model-native teacher prompt rendering with exact student completion replay,
  plus XGrammar-constrained student, teacher, and current-policy token
  probabilities with per-position alignment evidence;
- IW-OPD support for sparse environment masks, cached rollout log-probabilities,
  LoRA-only vLLM synchronization, speculative configuration, and engine kwargs;
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
- TurboQuant and MTP are configured and qualified independently before their
  combination is promoted.

## Regression tests

Run the focused maintained-delta suite from this repository:

    uv run pytest -q \
      tests/test_constrained_replay.py \
      tests/test_vllm_generation.py \
      tests/test_dapo_dynamic_sampling.py \
      tests/test_sampo_precomputed_advantages.py \
      tests/experimental/test_iw_opd_trainer.py

Run the GRPO selections that cover synchronization, parity, truncation,
advantages, and projection:

    uv run pytest -q tests/test_grpo_trainer.py \
      -k 'policy_parity or importance_sampling or truncated or advantage or logits_chunking'

Then run Ruff, the complete TRL test suite, and the Posttrain adapter contract
tests. A package release additionally requires a clean install from the exact
built artifact and GPU canaries for the selected DAPO and IW-OPD profiles.

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
2. build `trl==1.9.2.post5` from the immutable release commit;
3. record wheel and source hashes;
4. create and verify the immutable CarbonTeq tag and release;
5. upload the exact artifacts to the internal stable index;
6. verify a clean install imports the expected version and IW-OPD API;
7. update the Posttrain dependency, lockfile, fork consumer page, and release
   receipt;
8. run Posttrain CPU contracts and the selected GPU canaries before promoting a
   Posttrain release.
