# CarbonTeq TRL fork ledger

This ledger records the maintained, generally reusable delta between
`carbonteq-ai/trl` and `huggingface/trl`. Model selections, environment policy,
job configuration, and qualification evidence remain in the consuming
Posttrain framework.

Fork status: `candidate`, version `1.12.0.post5`, complete-group admission.
Published development candidate source: `b9f3a09369d9cfa21950feef3e110e1fdf779c54`,
tag `carbonteq-v1.12.0.post5`. Retained-asset publisher `34224729623` passed
clean installation and exact-byte readback. Wheel SHA-256:
`1f42571c28e178bb292eb7b904940f0d0e4b0ccdf936b23b8bec9704190b6ecb`;
sdist: `ceb581cc5a3d7a4a8a34cbc1b7fbc64e7aba9e3c55a6508b4a14ce257d341bc9`.
Stable promotion remains pending GPU qualification.
Post5 consumes validated `retained_input_indices` from external rollouts before
reward calculation, preserving source identity and whole groups. Ordinary
single-process GRPO pads only scored tensors to its scheduled accumulation
window with zero credit and retained-row sequence-mean normalization. Active
sampling accepts partial/empty candidate rounds within its existing bound.
No rewards or native episodes are invented. Partial multi-process, multimodal,
fixed alternate objectives, fused loss, entropy bonus and auxiliary loss paths
remain unsupported. Files: `trl/trainer/rollout_admission.py`,
`trl/trainer/grpo_trainer.py`, `tests/test_rollout_admission.py`. Regression
coverage includes two actual trainer updates for GRPO and active sampling,
source-row identity, empty candidate rounds and gradients at microbatch 1/2/4.
Run `python -m pytest tests/test_rollout_admission.py tests/test_olmo3_grpo_config.py`.
The consuming GPU qualification remains open; do not infer throughput fixes.
Previous post4 commit: `19e6c89a18617f1bd6e6385212705a67f5434962`.
Post4 source branch: `codex/trl-parity-probe-bound`. It includes the trainer
initializer plumbing required by the post3 configuration surface; publication
and live GPU qualification remain open. Previous post3/post2 releases follow.
Post2 release commit: `95a787b6c04f91a5d485fd827d31b1e1fb67ae8e`.
Tag: `carbonteq-v1.12.0.post2` (retained GitHub prerelease).
Wheel SHA-256: `8b19cd7fad5a28cf9ced4a7fac93bfaaedaac154faf019a3d3dd6abeaaa45c26`.
Source SHA-256: `0f6407d8280d59b433ef28ed5d007ddde95bb064cd28f12661fa8e59ef2e67c9`.
GitHub asset digests match both build-once artifacts. Installed post2 wheel
passes CUDA native IW-OPD train/resume/export at accumulation 1 and 2.
Development publication: Posttrain run `34007394648` succeeded, including
clean installation and retained-byte readback. Stable promotion remains open.
Sixty-six IW-OPD tests passed before expanding the skip matrix; the expanded
map/streaming x accumulation 1/2/3 x generations 1/2 matrix passes all 12 cases.

Previous development candidate (not promoted): `1.12.0.post1`.
Release commit: `6a5532e2f51e4e1cdc8a891582514a50f68a775a`.
Tag: `carbonteq-v1.12.0.post1`.
Wheel SHA-256: `cf242fafdfe476b7b8a250b300d6cbd52f502410a4053f4a9bac3366287727c5`.
Source SHA-256: `1e7bae5ee846972be7763e66dbd0b1149d99fb44c51125e2499adf4773d2e127`.
Both hashes match GitHub's asset digests. Forty-five retained-feature tests pass
against the installed wheel, independent of source-checkout imports.
Development publisher: Posttrain Actions run `34006220244`; stable promotion
and runtime-image qualification remain open. The maintained
features from consumer pin `69cf80a7319079ec5523841553467e119ebc1cec` are integrated
on upstream v1.12.0. Current branch: `codex/trl-1.12-retained-features`.
Historical publication record: `trl==1.9.2.post8` was a published CarbonTeq
build, tagged at commit `9a219ce5a593d85fe6058025de211ce42267e6b6` as
`carbonteq-v1.9.2.post8`. Its exact wheel SHA-256 is
`9933755547f3d09e5abef195607e5d8b682d277a4fc1febce6bd2cd053140ca5` and its
source-distribution SHA-256 is
`c6e167ffafae776e5c9514e9a8777058c0c05835afe568caf4e063df962bb06c`.

## Upstream base

- Upstream repository: `https://github.com/huggingface/trl`
- Upstream base: `59c4a8e104413fa9f4ca1a54eaf2ff93c0f299be` (`v1.12.0`)
- CarbonTeq repository: `https://github.com/carbonteq-ai/trl`
- Historical release branch: `v1.9.2.post5-release` (published `1.9.2.post11`; the
  already-reserved `carbonteq-v1.9.2.post5` through `post7` tags name
  unrelated commits)
- Historical package release: `trl==1.9.2.post8`
- Historical release commit: `9a219ce5a593d85fe6058025de211ce42267e6b6`

The Posttrain dependency declaration and lockfile are the executable consumer
authority. A candidate capability is not published until its fork commit,
immutable release tag, package hashes, and clean-install verification exist.

## Maintained delta

### Asynchronous rollout session foundation (2026-09-08)

`trl/generation/async_vllm_session.py` introduces a narrow, trainer-owned
per-request vLLM session. It fences one synchronized policy version, admits
independent asynchronous requests, explicitly aborts and drains them, and only
then allows inference residency to sleep for an optimizer update. The session
does not own environments, rewards, worker processes, or actor computation.
It is an additive foundation for the Posttrain rollout-execution workstream;
it is not yet connected to a trainer's vLLM construction path.
`tests/test_async_vllm_session.py` proves independent completion,
policy-version fencing, abort, drain, sleep/wake ordering, and idempotent
shutdown with a deterministic async engine double. The selected vLLM 0.25.1
surface is checked by `tests/test_async_vllm_runtime_contract.py`.
`scripts/qualify_async_vllm_lifecycle.py` is the bounded real-engine gate. On
2026-09-08 it passed locally with `Qwen/Qwen2.5-0.5B-Instruct`, two resident
sequences, a 512-token context, and vLLM 0.25.1: one request was explicitly
cancelled, two collection rounds completed with sampled-token logprobs, and
the engine drained, slept, restored weights and KV cache in separate stages,
and shut down cleanly. This exposed and fixed two native lifecycle details:
the initial resident engine must not be woken, and a weights-only wake leaves
request scheduling paused until the KV cache is restored. Actor-to-engine
changed-weight parity remains open, so the consumer must retain the existing
actor/sampler parity gate before this can be promoted.

### Async trainer model-request drain (2026-09-08)

The experimental async GRPO and async distillation trainers now use a
two-phase rollout-worker notification around weight publication. Before vLLM
is paused, `prepare_model_update(next_version)` atomically closes admission for
new model requests and waits for already-admitted requests to finish. An
episode executing a tool is not cancelled or replayed; its next model request
waits at the closed gate. After weight transfer and vLLM resume,
`update_model_version(version)` publishes the new version and reopens request
admission. Top-level group admission is closed over the same interval.

The shared active-request counter and admission condition make the drain
acknowledged rather than a best-effort event check: a request cannot race from
the child process into vLLM after the trainer has observed a completed drain.
The trainer continues to own update timing and stale-sample policy. The worker
does not abort tool calls, discard samples, compute importance weights, or
change reward semantics. Equivalent lifecycle changes are retained in both
experimental trainers to avoid divergent update behavior.

`tests/experimental/test_async_grpo_trainer.py` proves the exact
prepare/pause/transfer/resume/publish order and that preparation blocks until
an admitted model request acknowledges completion. It also proves that a
weight-transfer exception leaves the prior model version authoritative and
never resumes inference or publishes/reopens the pending version; the error is
run-fatal and propagates to the trainer. The broader experimental
trainer suite remains subject to its optional Flash Attention `kernels`
runtime dependency. This local candidate is not part of the published post5
package and has not passed changed-weight GPU qualification.

### Async changed-weight parity gate (2026-09-09)

`scripts/qualify_async_vllm_changed_weight.py` is the bounded native NCCL
release gate left open by the lifecycle probe. It assigns distinct visible
trainer and inference GPUs, teacher-forces one actor-selected token before and
after changing the final normalization weight, sends the changed tensor through
vLLM's native weight-transfer engine, resets the prefix cache, and requires the
changed sampler log probability to match the reference actor within an
explicit tolerance. It fails before engine construction unless two distinct
GPUs are available.

Two exploratory single-GPU attempts on the local RTX 3070 Ti were rejected and
are not qualification evidence. The first confirmed that AsyncLLM's process
boundary does not support sending an arbitrary callable through the frontend;
the second used the supported NCCL transfer API and NCCL rejected assigning
both ranks to one device. The probe now uses only the native NCCL path and
requires a two-GPU job. No changed-weight parity result has passed yet.

### Custom async worker consumption and recovery seam (2026-09-09)

The experimental async GRPO trainer now offers optional, backend-neutral hooks
for custom rollout workers to receive the group identity of every sample
admitted into a learner microbatch and to save or restore JSON scheduling state
with the trainer checkpoint. The acknowledgement occurs at learner collation,
not generation or queue publication, so an external task cursor does not move
merely because speculative work was produced. Duplicate group identities are
preserved because one microbatch may consume multiple siblings.

The trainer persists only the custom worker's declared scheduling metadata.
It does not serialize queues, environments, model requests, subprocesses, or
other live runtime state. Existing `AsyncRolloutWorker` prompt-index recovery
remains unchanged, and workers that do not implement the optional hooks retain
their prior behavior. Focused tests cover per-sample acknowledgement and custom
state save/load ordering. The Posttrain consumer must still enforce whole-group
checkpoint boundaries for algorithms whose replay cannot tolerate a partially
consumed group; this generic hook does not invent that algorithm policy.

### Stable-base integration (2026-09-06)

The post1 CUDA lifecycle gate exposed a checkpoint recovery failure in the
upstream IW-OPD `_RepeatBatchDataLoader` optimization: Accelerate reconstructs
an ordinary CPU loader, losing both prepared-device placement and accumulation
repetition. Post2 keeps repetition in `RepeatSampler` or the existing iterable
dataset helper and returns the native prepared loader, matching the maintained
DistillationTrainer pattern. This trades repeated collation for correct resume;
generation still happens once per optimizer window. Map and streaming loader
regressions cover accumulation 1/2/3 and checkpoint skipping. Native tiny-model
CUDA IW-OPD at accumulation 1 and 2 completes two nonzero-gradient updates,
resumes checkpoint 1 to matching final weights, and exports/reloads/generates.
This is not a vLLM, LoRA, multi-rank GPU, or model-quality qualification.

The upstream generation-window correction supersedes our Liger 0.8.0 adaptation
at `7fe16760`. The new candidate requires Liger >=0.8.2 and passes the full
window count through the native kernel API. It does not normalize twice.
The same numerical and two-rank gradient regressions are retained; only their
mock kernel follows the new argument contract.

The new upstream vLLM weight-transfer iterator/packed server API remains intact.
Weight namespace prefixes are applied before metadata and tensor iteration;
LoRA-only sync still exports only the adapter. Full-weight level-2 wake restores
the current actor, never the initial checkpoint through `reload_weights`.
Bounded request waves, raw parity probes, exact-token IW-OPD, sampling controls,
active sampling, memory-bounded projection, and MTP/KV metrics remain maintained.
The obsolete `use_binary` server argument was removed from the raw parity probe.

Upstream's NaN sampling-logprob exclusion is combined with our globally
token-weighted metrics. Upstream's differentiable entropy bonus must remain
differentiable in both chunked and ordinary scoring. Consumer state is in
`docs/tooling/trl/README.md` and `docs/plan/gdpo-capo-dual-backend-support.md`.

The fork keeps the following behavior on top of upstream 1.12.0:

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
- IW-OPD recomputes `num_items_in_batch` from buffered completion labels after
  on-policy generation. The base Trainer count observes the prompt-only raw
  batch and is zero for a fully on-policy window; carrying the post-generation
  count prevents a finite token-loss sum from being divided by zero;
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

The one-time raw-parity probe also has an independent prompt-plus-completion
sequence bound. It left-truncates only the probe prompt and sends that exact
bounded context to both vLLM and the actor. Actor recomputation runs only over
the selected probe rows, rather than the full padded rollout batch. This keeps
the synchronization gate meaningful while preventing a safety check from
materializing an unrelated production-sized actor forward; rollout and update
sequences are unchanged.

## Compatibility constraints

- Package baseline: Python 3.10+, Transformers as declared by upstream 1.12.0,
  and vLLM `>=0.19.0,<=0.27.1`. Broader upstream support does not extend
  Posttrain's model/runtime qualification beyond its selected 0.25.1 runtime.
- Posttrain qualification target: Python 3.13, Torch 2.11, Transformers 5.14,
  vLLM 0.25.1, PEFT 0.19, Liger >=0.8.2, and `datasets>=4.6.1,<4.7`.
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

Stable-base candidate evidence (2026-09-06): 106 tests passed across
`test_liger_window_normalization.py`, `test_vllm_generation.py`,
`test_dapo_dynamic_sampling.py`, `test_sampo_precomputed_advantages.py`, and
`experimental/test_iw_opd_trainer.py`. The latter includes native trainer
execution and post-generation item-count/finite-loss regressions. Fourteen
additional GRPO parity/projection/truncation tests passed. Twenty config,
version-boundary, and client parsing tests are retained. Framework adapter
checks: 26 passed against this source. These do not replace the real colocated
vLLM/runtime-image qualification gate or establish publication.

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

The bounded-sequence parity correction is covered by
`test_vllm_policy_parity_probe_bounds_each_sequence_and_left_truncates_prompt`
and the associated configuration validation in `tests/test_grpo_trainer.py`.

Run the changed-weight gate only on a job with two distinct visible GPUs:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/qualify_async_vllm_changed_weight.py \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --trainer-device 0 --inference-device 1 \
      --max-model-len 512 --gpu-memory-utilization 0.45

Then run Ruff, the complete TRL test suite, and the Posttrain adapter contract
tests. Publication is manual from Posttrain's repository-scoped retained-asset
candidate workflow: it downloads the immutable GitHub Release assets, verifies
their supplied SHA-256 values, uploads the exact bytes only to
`carbonteq/dev`, proves development-channel readback, and retains a clean
install receipt. After Posttrain candidate qualification, its separate
repository-owned promotion workflow re-verifies those same bytes and transfers
them server-side to `carbonteq/stable`. Forks do not receive release runners
or execute fork-controlled publication workflows.

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
5. use Posttrain's manual retained-asset candidate workflow to upload the
   exact artifacts only to `carbonteq/dev` and prove their readback;
6. verify a clean development-index install imports the expected version and
   IW-OPD API;
7. materialize the Posttrain candidate lock and run its CPU contracts and
   selected GPU canaries;
8. use Posttrain's separate promotion workflow to move the unchanged,
   hash-verified artifacts to `carbonteq/stable`;
9. update the Posttrain stable dependency, lockfile, fork consumer page, and
   release receipt.
