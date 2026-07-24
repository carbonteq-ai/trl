# CarbonTeq TRL fork ledger

This file records the maintained delta between `carbonteq-ai/trl` and
`huggingface/trl`. Generic trainer and runtime changes belong here; project
model selections, environment policy, and qualification evidence belong in the
consuming post-training framework.

## Upstream base

- Upstream repository: `git@github.com:huggingface/trl.git`
- Upstream base: `95809b942eb5d11d0b06d749510d88be99230b73` (`Release: v1.8 (#6346)`)
- CarbonTeq remote: `git@github.com:carbonteq-ai/trl.git`
- Published implementation commit: `76dd120e88437cfa44c27fc5b17f4fed68ebfd91`
- Current development branch: `codex/dapo-dynamic-sampling`

The post-training framework's executable pin is in
`../rl/packages/train/pyproject.toml` and `../rl/uv.lock`. Do not describe a
working-tree change as published support or update that immutable pin before
the fork commit is pushed.

## Maintained delta

The fork currently maintains:

- vLLM 0.24 and 0.25 dependency compatibility;
- `datasets 4.6` compatibility for Verifiers integration;
- memory-efficient entropy metrics for non-contiguous slices;
- colocated GRPO vLLM speculative configuration and guarded engine kwargs;
- model weight-name prefixes, native LoRA synchronization, and compatible
  sleep/wake behavior for quantized bases;
- exact-token Verifiers rollout hooks and dataset identity for GRPO and
  experimental on-policy distillation;
- candidate support for native MTP and KV-cache dtypes in on-policy
  distillation, plus normalized per-generation speculative-decoding metrics;
- a candidate vLLM 0.25.1 TurboQuant cache-marker compatibility guard which
  activates only when the installed build still reports no TurboQuant
  quantization mode;
- an optional GRPO log-probability projection chunk size in
  `trl/trainer/grpo_config.py`, with the chunked LM-head projection in
  `trl/trainer/grpo_trainer.py` and numerical-equivalence coverage in
  `tests/test_grpo_trainer.py`;
- bounded DAPO dynamic sampling in `GRPOTrainer`, which retains informative
  prompt groups, refills only missing groups from sequential candidate batches,
  recomputes the global token normalizer after filtering, and refuses partial
  training batches;
- an opt-in `rollout_func` contract for finite, token-aligned precomputed
  advantages. This supports hierarchical agentic estimators without moving
  environment or algorithm ownership into TRL.

Native MTP here means rollout acceleration through the model's bundled draft
head. It does not add an MTP auxiliary training loss. TurboQuant changes only
the rollout KV cache; it does not quantize trainable actor weights.

The chunked projection keeps hidden states and the frozen LM head semantically
unchanged while avoiding one full `[batch, sequence, vocabulary]` logits
allocation during old-policy and reference-policy scoring. A positive
`logits_chunk_size` selects the maximum number of flattened token positions per
projection. `None` retains upstream's unchunked behavior. This is a generic
memory-scheduling control: it does not select a model, environment, task,
prompt budget, GRPO group size, or tracking provider.

## Compatibility constraints

The candidate MTP and TurboQuant additions apply to colocated vLLM engines.
External server mode must receive equivalent options when the server process is
launched. The shared vLLM constructor rejects engine kwargs which attempt to
override TRL-owned model, lifecycle, synchronization, or sampling arguments.

The current consumer resolves Python 3.12, Torch 2.11, Transformers 5.14, and
vLLM 0.25.1. On the local Ampere GPU, TurboQuant K8V4 requires an FP16 rollout
copy. The consumer adapter selects that dtype when K8V4 is requested.

Chunked projection requires the same token-aligned masks and temperature
handling as the unchunked path. Its regression test compares both
log-probabilities and entropies. It reduces projection peak memory, but it does
not make the differentiable GRPO loss memory-bounded by itself. The consuming
framework currently combines it with Liger's fused GRPO loss for the
single-GPU profile. Liger remains a consumer/runtime choice rather than a fork
default.

DAPO dynamic sampling is opt-in and text-only. Each process must own complete
prompt groups so selection never splits a reward-normalization group across
ranks. `dynamic_sampling_max_batches` bounds rollout work; exhausting it raises
instead of silently changing the optimizer batch size. The implementation does
not incorporate Dr. GRPO, GSPO, or mixed-policy guidance under the DAPO name;
those methods use different estimators or policy sources.

`use_precomputed_advantages=True` requires `rollout_func`, rejects Liger, and
requires one finite advantage value per completion token. Reward computation
still runs for filtering and evidence; only the policy-loss advantage is
replaced. Dynamic sampling may use this path with the ordinary clipped loss,
while ordinary GRPO cannot enable dynamic sampling accidentally.

TurboQuant is configuration-supported but not Qwen 3.5 quality-qualified: its
32K matched recall gate remains open. MTP and TurboQuant must be qualified
independently before testing their combination.

The consuming framework qualified the candidate native-MTP GRPO path with
Qwen 3.5 0.8B at a 32K engine window on an 8 GiB GPU. The two-step run used
physical microbatch one plus gradient accumulation two, completed four original
AutomationBench trajectories, synchronized the post-update LoRA adapter, and
recorded non-zero MTP acceptance in both steps. A follow-up run verified that
turn-local vLLM counters are summed into step totals and rates are recomputed
from those totals. This evidence does not qualify on-policy distillation or
TurboQuant quality.

## Regression tests

From this repository, run:

    uv run pytest tests/test_vllm_generation.py tests/test_grpo_trainer.py \
      tests/experimental/test_distillation_trainer.py

For the focused candidate delta, run:

    uv run pytest tests/test_vllm_generation.py tests/test_grpo_trainer.py \
      tests/experimental/test_distillation_trainer.py \
      -k 'speculative or vllm_boundary or lora or colocated_vllm_engine_options'

For the chunked projection specifically, run:

    uv run pytest tests/test_grpo_trainer.py \
      -k test_logits_chunking_matches_unchunked_logprobs_and_entropy

For DAPO dynamic sampling specifically, run:

    uv run pytest tests/test_dapo_dynamic_sampling.py

For token-aligned agentic advantages specifically, run:

    uv run pytest tests/test_sampo_precomputed_advantages.py

Run Ruff and the repository's standard test suite before publishing. The
developer environment does not include vLLM by default; validate the guarded
TurboQuant marker in the consumer's pinned vLLM 0.25.1 environment as a release
gate.

The chunked and unchunked projections have passed the focused numerical
equivalence test. The consumer has also crossed the previously failing
large-vocabulary scoring allocation on an 8 GiB GPU with chunk size 128 and a
fused differentiable loss. A complete multi-update GPU run is still required
before that operating profile is called qualified.

## Rebase procedure

Fetch `upstream`, identify the new immutable upstream base, and create a fresh
`codex/` branch. Reapply the maintained deltas in small groups, starting with
dependency compatibility, then shared vLLM lifecycle changes, then trainer
hooks. Run the focused tests after each group and the full suite at the end.
Check whether upstream now exposes equivalent speculative configuration,
engine kwargs, distillation hooks, MTP telemetry, a native TurboQuant mode, or
an equivalent memory-bounded GRPO projection; drop local changes whose behavior
is upstream and retain regression coverage.
Update this base, delta, constraints, and published commit only after the new
branch is validated and pushed.

## Publication checklist

Before updating a consumer pin, record a clean fork commit, push it to the
CarbonTeq remote, run the focused regression suite, run the consumer adapter
tests, and execute the relevant GPU qualification. The consumer page at
`../rl/docs/tooling/trl/README.md` owns the selected models, operating values,
run evidence, and outstanding release gates.
