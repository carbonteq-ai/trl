# CarbonTeq TRL fork ledger

This ledger records the maintained, generally reusable delta between
`carbonteq-ai/trl` and `huggingface/trl`. Model selections, environment policy,
job configuration, and qualification evidence remain in the consuming
Posttrain framework.

Fork status: `candidate`, version `1.12.0.post6`.
Post6 consolidates the asynchronous rollout-session lifecycle, changed-weight
transfer fencing, checkpointed custom rollout scheduling, bounded agent
collection, and complete-group acknowledgement described below. Its immutable
release commit, tag, artifact hashes, and development-channel readback are
recorded only after the build-once publication gates pass. The test extras now
declare `pytest-asyncio`, so a clean contributor environment can execute the
async session regressions without an undeclared local dependency.

The last published development candidate is post5, source
`b9f3a09369d9cfa21950feef3e110e1fdf779c54`,
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
package; the changed-weight gate below qualifies its transfer boundary but not
a complete optimizer update or package publication.

### Async changed-weight parity gate (2026-09-09)

`scripts/qualify_async_vllm_changed_weight.py` is the bounded native NCCL
release gate left open by the lifecycle probe. It runs on a trainer GPU against
a separately launched `vllm serve` process, teacher-forces one actor-selected
token before and after changing the final normalization weight, sends the
changed tensor through async GRPO's production HTTP/NCCL clients, resets the
prefix cache, and requires the changed server log probability to match the
reference actor within an explicit tolerance. The server may be on another
host when both hosts expose mutually routable addresses. The accompanying
dstack task describes a two-node on-demand shape without embedding a local
checkout or machine-specific path.

The async control client now checks every HTTP response and uses explicit
timeouts. A rejected pause, resume, introspection, or weight-update request is
an exception rather than a false acknowledgement; focused client tests cover
this fail-closed boundary. The clients use ordinary Python logging so they can
run in a standalone qualification process without first initializing
Accelerate state. The shared vLLM control client also accepts a successful
empty response body, as returned by the server's prefix-cache reset endpoint,
instead of attempting to decode it as JSON.

Two exploratory single-GPU attempts on the local RTX 3070 Ti were rejected and
are not qualification evidence. The first confirmed that AsyncLLM's process
boundary does not support sending an arbitrary callable through the frontend;
the second used the supported NCCL transfer API and NCCL rejected assigning
both ranks to one device. The obsolete in-process topology was then replaced
by the production external-server topology. A two-node RunPod plan parsed
successfully on 2026-09-09 but had no matching clustered offer and was not
submitted, so it incurred no cloud workload.

The external-server gate passed on 2026-09-09 with the candidate source at
`2308ab41aeedc082e154734f33cfe44809b1fdea`. The actor ran on an RTX 3070 Ti
and the server ran on an RTX PRO 6000 Blackwell Workstation Edition using
`registry.carbonteq.com/carbonteq/posttrain-kind-online-rl-trl-py312@sha256:8230413ea572158e59e3f4099b218474d339869fb3eb1676ebaf23e35d35d03d`
with vLLM 0.25.1, Torch 2.11.0+cu130, and Transformers 5.14.1. Base
actor/server log-probability delta was `0.0022419691`; after transferring the
changed `model.norm.weight` tensor the delta was exactly `0.0`, while the
server log probability itself moved by `10.6749088764`. This proves both
numerical parity and that the server used the transferred weight rather than
returning an unchanged cached result.

The immutable slim runtime has no CUDA compiler. Its server therefore sets
`VLLM_USE_FLASHINFER_SAMPLER=0` explicitly. This does not disable FlashInfer
attention or NCCL transfer, and vLLM 0.25.1 already requires the native sampler
for this gate's `processed_logprobs` mode because FlashInfer sampling cannot
return post-top-k/top-p log probabilities. Production rollout-only sampling
must still benchmark the FlashInfer path in a runtime with its kernels
available; this qualification result is not evidence that disabling the
sampler is throughput-neutral for other modes.

`scripts/qualify_async_vllm_failure_boundary.py` exercises the corresponding
live fail-closed path against the same external server. After initializing the
real NCCL group, it deliberately requests `finish_weight_update` before
`start_weight_update`. On 2026-09-09 vLLM rejected the transition with HTTP
500; TRL prepared version 1 but published no version, its authoritative model
version remained 0, and the server's selected-token log probability was
exactly unchanged at `-1.2563054562` after explicit qualification cleanup.
This closes the single-rank live control-failure gate. It does not qualify
distributed multi-rank failure propagation, an optimizer update, or resume.

The qualification server is disposable. vLLM 0.25.1 retains an initialized
NCCL transfer group for the server lifetime, so a new probe process must use a
fresh server process. A failed harness attempt also showed that the native
client's server-initialization request runs in a background thread while the
trainer joins the NCCL store; if the server rejects initialization immediately,
the trainer can wait until the store timeout. Production initializes this group
once per server, but bounded initialization-failure propagation remains an
explicit distributed-runtime gate rather than a claimed result.

### Custom async worker consumption and recovery seam (2026-09-09)

The experimental async GRPO trainer now offers optional, backend-neutral hooks
for custom rollout workers to receive the group identity of every sample
actually dispatched to `training_step` and to save or restore JSON scheduling
state with the trainer checkpoint. The collator only transports group IDs:
Hugging Face dataloaders may prefetch and collate a later microbatch before the
learner uses it, so acknowledging in the collator incorrectly advances an
external task cursor. Duplicate group identities are preserved because one
microbatch may consume multiple siblings.

The trainer persists only the custom worker's declared scheduling metadata.
It does not serialize queues, environments, model requests, subprocesses, or
other live runtime state. Existing `AsyncRolloutWorker` prompt-index recovery
remains unchanged, and workers that do not implement the optional hooks retain
their prior behavior. Focused tests cover per-sample acknowledgement and custom
state save/load ordering. The Posttrain consumer must still enforce whole-group
checkpoint boundaries for algorithms whose replay cannot tolerate a partially
consumed group; this generic hook does not invent that algorithm policy.

### Native async agent rollout qualification (2026-09-09)

`scripts/qualify_async_agent_rollouts.py` exercises the original
`AsyncRolloutWorker` rather than Posttrain's custom producer. A local
Qwen3.5-2B vLLM server completed 16 groups of four trajectories with 64/64
successful `record_choice` tool calls and zero tool failures. With the same
model, prompt set, sampling controls, and 128-token cap, producer/server
concurrency 32 completed in 18.7395 seconds (3.4152 samples/s), versus 46.7045
seconds (1.3703 samples/s) at concurrency one: a 2.49x throughput increase and
27.965 seconds, or 59.9%, less wall time for this bounded workload.

The first live attempt exposed two standalone-lifecycle defects now covered by
regressions: parent worker logging depended on pre-existing Accelerate state,
and normal shutdown reported deliberately cancelled generation tasks as worker
failures under Python 3.13. Worker lifecycle logging is now independent of the
trainer and cancellation is suppressed only when the stop event is set; an
unexpected cancellation still propagates. This gate establishes tool-capable
async rollout behavior and concurrency benefit on one RTX 3070 Ti. It does not
establish a 2B learner optimizer update or linear scaling to 32 requests.

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

Launch a vLLM server on one GPU:

    VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
      --host 0.0.0.0 --port 8000 --dtype half --enforce-eager \
      --max-model-len 512 --max-num-seqs 2 \
      --weight-transfer-config '{"backend":"nccl"}' \
      --logprobs-mode processed_logprobs

Then run the gate on a distinct trainer GPU that can reach the server and can
itself be reached by the server's NCCL rank:

    python scripts/qualify_async_vllm_changed_weight.py \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --server-url http://SERVER_IP:8000 --trainer-device cuda:0

Use a fresh server process, then run the live fail-closed gate from the trainer
GPU:

    python scripts/qualify_async_vllm_failure_boundary.py \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --server-url http://SERVER_IP:8000 --trainer-device cuda:0

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
