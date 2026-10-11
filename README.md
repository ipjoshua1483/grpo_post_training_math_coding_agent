# GRPO Post-Training with RLVR for Math and Coding Multi-Turn LLM Agents

An LLM agent that solves math and Python function-generation problems, optionally runs Python in a restricted Docker sandbox, and learns from binary correctness rewards through TRL GRPO. The base model is **Qwen2.5-1.5B-Instruct**.

This repository contains the core implementation, the three reported experiment conditions, fixed evaluation IDs, compact evidence, and a results generator. It excludes checkpoints, downloaded datasets, full model transcripts, and superseded experiments.

## Three experiments

| ID | Model | Prompt and interaction | Purpose |
|---|---|---|---|
| **E1: Few-shot base** | Untrained instruct model | Detailed domain-specific prefixes with synthetic workflow examples; optional Python tool; explicit textual controller | Prompted baseline |
| **E2: Compact base** | Same untrained model | Short math prefix and its coding adaptation; optional tool; no few-shot examples | Pre-RL comparison |
| **E3: Compact + GRPO** | Checkpoint at optimizer step 25 | Exactly the E2 prompts and evaluation policies | Post-training comparison |

Both baselines allow tools. Their observed zero-call behavior in some domains does not make them deliberately tools-disabled experiments. No required-tool condition is part of these reported results.

The exact prefixes are easy to inspect in [prompts/](prompts/): `fewshot-math.txt`, `fewshot-code.txt`, `compact-math.txt`, and `compact-code.txt`. Their JSON counterparts preserve message roles and few-shot turns. The model-specific rendering is implemented in [runtime.py](src/tool_rl/runtime.py).

**One comparison caveat:** E2 math also supplies the tool schema to Qwen's chat template and returns observations with the native `tool` role. E1 math uses explicit textual feedback messages. Both coding conditions use the explicit textual controller. E1 versus E2 therefore compares complete prompt/interaction conditions; it does not isolate the causal effect of removing examples. E2 versus E3 keeps that interface fixed.

## Results

One greedy answer per problem, seed **42**, on fixed held-out subsets. Math uses accuracy; coding uses pass@1 against the dataset's original tests.

| Benchmark | Examples | E1: Few-shot base | E2: Compact base | E3: GRPO step 25 | E3 − E2 |
|---|---:|---:|---:|---:|---:|
| GSM8K test | 200 | 100/200 (**50.0%**) | 104/200 (**52.0%**) | 110/200 (**55.0%**) | **+3.0 pp** |
| MATH-500 | 200 | 79/200 (**39.5%**) | 70/200 (**35.0%**) | 74/200 (**37.0%**) | **+2.0 pp** |
| HumanEval | 50 | 21/50 (**42.0%**) | 17/50 (**34.0%**) | 20/50 (**40.0%**) | **+6.0 pp** |
| MBPP test | 100 | 32/100 (**32.0%**) | 27/100 (**27.0%**) | 33/100 (**33.0%**) | **+6.0 pp** |

Post-training improves all four scores relative to the matched compact baseline. Results use fixed held-out subsets and a single seed.

The fraction of problems with at least one tool request was:

| Benchmark | E1 | E2 | E3 |
|---|---:|---:|---:|
| GSM8K | 0% | 48.0% | 42.0% |
| MATH-500 | 0% | 28.0% | 24.5% |
| HumanEval | 0% | 0% | 0% |
| MBPP | 0% | 0% | 0% |

Tool requests were observed in math; coding responses remained direct in all three conditions. Request rates include invalid calls; execution counts are available in the numerical reports.

## Evidence you can regenerate

| Evidence | Included | What can be checked |
|---|---|---|
| E1 evaluation | Historical aggregate scores and tool-call rates | Arithmetic and reported table; per-example outcomes unavailable |
| E2 evaluation | 550 per-example metric rows | Scores, IDs, tool counters, lengths, truncation, format, errors |
| E3 evaluation | Historical aggregate summary | Arithmetic and reported table; per-example outcomes unavailable |
| Training | Reward/ID/tool counters for all ranks; rank-0 TRL metrics and audits | Completed-update reward groups and gradient/update evidence |

The reporter marks E1/E3 as `aggregate_only`; it never reconstructs nonexistent trajectories. The intended common evaluation IDs are retained in [eval-controls.json](repro/eval-controls.json). E2's shipped logs can be checked against them; original E1/E3 evaluation manifests are unavailable locally, so their ID identity cannot be independently rechecked from per-example evidence.

Model tokens, gold answers, problem text, and hidden tests have been stripped from the exported metric logs. [provenance.json](results/provenance.json) records evidence gaps and source-log hashes. [training-run.json](repro/training-run.json) records the selected training IDs, configuration, and file hashes. [public-snapshot.json](repro/public-snapshot.json) records this snapshot's source hashes and CPU validation; no new GPU experiment was run while preparing it.

Generate the tables and training statistics without a GPU or Docker:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python report_results.py --check-recorded --output results/generated
python -m unittest discover -s tests
```

On Windows, activate with `.\.venv\Scripts\Activate.ps1`. The CPU test suite covers parsing/recovery, extraction, hidden-test separation, log aggregation, masks, and fake-backend inference. An optional full-dataset test is skipped until frozen data is installed. These tests do not run a real model or assert that the local Docker engine is available.

Outputs are `summary.json`, `scores.csv`, and `comparison.md`.

If additional original evaluation logs become available, supply **merged** metric files, once per domain:

```bash
python report_results.py \
  --run E3=/path/to/post-math/metrics.jsonl \
  --run E3=/path/to/post-code/metrics.jsonl \
  --check-recorded --output results/regenerated
```

The reporter rejects duplicate IDs, incomplete four-reward groups, and mismatched fixed evaluation IDs. `--require-raw` refuses aggregate fallbacks; it intentionally fails with the currently bundled E1/E3 evidence.

## Installation for inference and training

The tested GPU environment was Linux, Python 3.12, PyTorch 2.7.1/CUDA 12.8, Transformers 5.2.0, TRL 0.29.1, and Accelerate 1.12.0. [requirements/train.txt](requirements/train.txt) pins direct dependencies; [repro/environment.txt](repro/environment.txt) captures the recorded environment. CUDA inference is required by the current runtime; training uses all eight GPUs, while a single GPU suffices for inference.

From the repository root on a suitable GPU host:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/train.txt -e .
docker build -f docker/Dockerfile.sandbox -t tool-rl-python-sandbox:local .
docker build -f docker/Dockerfile.coding-sandbox -t tool-rl-coding-sandbox:local .
```

The host runs model inference/training; only generated code and symbolic/unit-test verification run in Docker. No GPU is exposed to the tool containers. Image IDs are recorded in new experiment manifests. The Python base-image tag is not digest-pinned, so rebuilding today may produce a different image from the historical run.

## Run inference on your own problem

Single-problem inference uses the same prompts, generation adapters, controllers, and extraction paths as evaluation. It supplies no gold answers or hidden tests, so output correctness is **not** certified.

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --domain math --prompt compact \
  --problem "What is the sum of the cubes of the integers from 1 through 25?" \
  --output runs/inference-math.json

CUDA_VISIBLE_DEVICES=0 python infer.py \
  --domain code --prompt compact --entry-point clamp_values \
  --problem "Implement clamp_values(values, lower, upper): return a new list with each numeric value clamped to the inclusive bounds. Assume lower <= upper." \
  --output runs/inference-code.json
```

Use `--prompt fewshot` for E1's prefix. `--problem-file problem.txt` accepts longer input. `--model /path/to/checkpoint-bf16` uses the separate BF16 inference checkpoint; model weights are excluded from this code repository. Base-model inference pins commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Tool use is optional, so a valid response can finish directly.

### Optional trained checkpoint

The step-25 model can be distributed as a separate **BF16 inference package**, approximately 3.1 GB, containing the weights, tokenizer, chat template, and configuration files. Optimizer, scheduler, and RNG state are unnecessary for inference and are excluded. Keep the package separate from GitHub source files.

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model /absolute/path/to/checkpoint-bf16 \
  --domain math --prompt compact \
  --problem "What is the sum of the squares from 1 through 30?"
```

To export an FP32 training checkpoint on CPU:

```bash
python scripts/export_bf16.py \
  --checkpoint /absolute/path/to/checkpoint-25 \
  --output /absolute/path/to/checkpoint-bf16
```

The export verifies every tensor against the BF16 conversion of the original weights. The evaluation runtime also loads weights as BF16. This export needs PyTorch and safetensors, but does not need CUDA or run model inference.

## Tool protocol, sandbox, and rewards

The only tool is `execute_python(code: str)`. The generated syntax is:

```text
<tool_call>
{"name":"execute_python","arguments":{"code":"print(23 * 17)"}}
</tool_call>
```

The explicit controller validates the JSON/name/argument type, executes the code, appends an observation such as the following, then generates again. A compatibility path also accepts a top-level `execute_python("literal source")` in a Python fence, parsed with AST rather than executed on the host:

```text
<tool_result>
{"ok":true,"stdout":"391\n","stderr":"","exit_code":0,"timed_out":false}
</tool_result>
```

For compact math, Qwen's native chat-template rendering uses the same textual call syntax; a local parser converts it to structured calls and returns the observation in a `tool` message. It does not depend on the tokenizer's missing `response_schema`. Invalid names or malformed JSON receive corrective feedback, consume only the invalid-call budget, and cannot trigger arbitrary tools.

| Budget | Math | Code |
|---|---:|---:|
| Valid execution requests | 2 | 4 |
| Invalid requests | 2 | 2 |
| Agent turns | 6 | 7 |
| Evaluation completion context | 1,536 tokens | 2,048 tokens |
| Prompt limit | 4,096 tokens | 4,096 tokens |

Every execution starts a fresh restricted CPU container: no network, no host mounts, no Docker socket, unprivileged UID, dropped capabilities, read-only root filesystem, disposable `/tmp`, one CPU, 256 MiB memory, 32 processes, 5-second CPU limit, 10-second wall limit, and bounded output. Observations are further trimmed before entering model context. Scratch code cannot access the benchmark grader's reference files. Docker shares the host kernel; this is container isolation rather than a VM boundary.

**Reward is binary final correctness.** Math uses exact rational/numeric normalization or bounded symbolic checking with Math-Verify 0.8.0. Coding runs the official HumanEval/MBPP tests, or curated KodCode training tests, in the coding container. All tests must pass for reward 1; partial test counts are diagnostic only. No bonus is awarded for tool use or final-tag formatting. Grading runs after the final candidate is extracted; scratch calls contain only model-written self-tests.

Math extraction accepts optional final tags, boxed answers, answer cues, and unambiguous final mathematical conclusions. Coding extraction accepts a final block, a Python fenced block, or an identifiable complete function. Truncated final submissions receive reward 0. The coding verifier is a custom restricted execution wrapper around the dataset tests; it does **not** claim EvalPlus extended-test coverage. A HumanEval `check(candidate)` suite counts as one diagnostic suite, despite containing multiple assertions.

## Rebuild held-out data and run evaluation

Dataset snapshots are pinned in [dataset-revisions.json](repro/dataset-revisions.json). GSM8K test, MATH-500, HumanEval, and MBPP test are evaluation-only. Math development examples are reserved from the official train splits. The initial math training subsets use GSM8K and MATH train; coding training uses a separate KodCode corpus.

```bash
python -m tool_rl.data \
  --config configs/data.json --output data/frozen \
  --revisions-from repro/dataset-revisions.json
python scripts/prepare_eval.py --data-dir data/frozen --output-dir eval-plans
```

The helper checks data hashes/revisions against the retained controls, preserves the exact historical IDs/order, and creates four plans with fingerprints for this public code snapshot. It refuses to overwrite plans. Namespace and packaging cleanup changed source fingerprints; old private plans should not be reused unchanged.

All evaluations use greedy decoding (`do_sample=False`), seed 42, and the fixed domain-specific budgets. Evaluation has one independent model replica per GPU, rank-based input sharding, separate rank logs, and a checked merge. It uses **no DDP gradient synchronization**. A math worker processes 50 total examples: 25 GSM8K and 25 MATH-500. Coding workers process 18 or 19 examples each.

Set these shell variables, then run each condition into fresh output directories:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct
REVISION=989aa7980e4cf806f80c7fef2b1adb7bc71aa306

# E1: detailed few-shot base
torchrun --standalone --nproc-per-node=8 evaluate.py \
  --model "$MODEL" --revision "$REVISION" \
  --plan eval-plans/fewshot-math.json --prompt-variant optional_tool \
  --config configs/eval-fewshot-math.json --output runs/E1-math
torchrun --standalone --nproc-per-node=8 evaluate.py \
  --model "$MODEL" --revision "$REVISION" \
  --plan eval-plans/fewshot-code.json --prompt-variant optional_tool \
  --config configs/eval-fewshot-code.json --output runs/E1-code

# E2: compact base
torchrun --standalone --nproc-per-node=8 evaluate.py \
  --model "$MODEL" --revision "$REVISION" \
  --plan eval-plans/compact-math.json --prompt-variant compact_math \
  --config configs/eval-compact-math.json --output runs/E2-math
torchrun --standalone --nproc-per-node=8 evaluate.py \
  --model "$MODEL" --revision "$REVISION" \
  --plan eval-plans/compact-code.json --prompt-variant compact_code \
  --config configs/eval-compact-code.json --output runs/E2-code

# E3: reuse the exact E2 plans/configs with the trained model
CKPT=/absolute/path/to/checkpoint-25
torchrun --standalone --nproc-per-node=8 evaluate.py \
  --model "$CKPT" --plan eval-plans/compact-math.json --prompt-variant compact_math \
  --config configs/eval-compact-math.json --output runs/E3-math
torchrun --standalone --nproc-per-node=8 evaluate.py \
  --model "$CKPT" --plan eval-plans/compact-code.json --prompt-variant compact_code \
  --config configs/eval-compact-code.json --output runs/E3-code
```

For one GPU, replace `torchrun --standalone --nproc-per-node=8` with `CUDA_VISIBLE_DEVICES=0 python`. Completed runs contain `summary.json`, merged `metrics.jsonl`, trajectories, and a manifest. A `--dry-run` validates plans and sharding without loading the model. Recomputed results can vary with rebuilt images or a different CUDA environment; the bundled evidence describes the recorded run.

## GRPO training

[train.py](src/tool_rl/train.py) reuses TRL's `GRPOTrainer`, supplying a multi-turn rollout function and masks that exclude tool observations from the policy loss. Advantages are normalized within each group of **four sampled completions for the same prompt**; an all-identical reward group has no reward-advantage signal. The KL term remains active.

| Setting | Recorded value |
|---|---|
| Selected pool | 64 GSM8K + 64 MATH + 128 KodCode = 256 problems |
| Hardware | 8 × H200 |
| Samples per prompt | 4 |
| Global completions per update | 8: one math and one coding group |
| Batch / GPU; accumulation | 1; 1 |
| Learning rate; KL coefficient | 1e-6; 0.02 |
| Sampling | Temperature 0.8, top-p 0.95, seed 42 |
| Loss / reward scaling | TRL `dapo` / `group`; one iteration |
| Precision / optimizer | FP32 trainable weights, BF16 mixed precision, AdamW |
| Completion-context limit | 1,536 tokens for both training domains |
| Logging / saving | Every 1 / 25 steps |
| Evaluated checkpoint | Step 25 |

The replayed token stream preserves generated assistant tokens and environment tokens. Environment tokens receive loss mask 0. Generated truncated completions remain in the token loss (`mask_truncated_completions=False`) while their verified reward is 0. The trainer records per-domain rewards and optimization diagnostics. Full model training uses DDP through Accelerate; evaluation does not.

To build a **new** training corpus and run the recorded configuration:

```bash
python -m tool_rl.joint_data \
  --base-data data/frozen --config configs/joint-data.json \
  --output data/joint-frozen
WORLD_SIZE=8 python train.py --config configs/grpo.json --data-dir data/joint-frozen --dry-run
torchrun --standalone --nproc-per-node=8 train.py \
  --config configs/grpo.json --data-dir data/joint-frozen \
  --revision "$REVISION" --output runs/grpo-main
```

Use `--steps 25` for a deliberately shorter new run. Output directories must be fresh; use `--resume-from-checkpoint` only with an existing compatible training run. Keep shell logs outside the trainer's output directory.

## Navigate the code

| Files | Responsibility |
|---|---|
| `infer.py`, `src/tool_rl/inference.py` | Public single-problem inference; no hidden grader |
| `evaluate.py`, `eval_cli.py`, `evaluate.py`, `coding_evaluate.py` under `src/tool_rl/` | Domain dispatch, GPU sharding, manifests, metrics, checked merge |
| `prompt_variants.py`, `protocol.py`, `coding_protocol.py`; `prompts/` | Exact compact/few-shot prefixes, examples, tool syntax |
| `runtime.py`, `qwen25.py` | Model/tokenizer loading, generation, native math adapter, token/mask preservation |
| `rollout.py`, `environment.py`, `coding_environment.py` | Multi-turn calls, feedback, independent budgets |
| `sandbox.py`, `coding_sandbox.py`, `worker.py`, `coding_worker.py`; `docker/` | Restricted execution and private final verification |
| `math_answers.py`, `rewards.py`, `coding_rewards.py` | Semantic extraction and binary verifiable rewards |
| `data.py`, `joint_data.py`, `records.py`, `experiment_data.py`, `coding_data.py` | Public/private splits, deterministic sampling, overlap filtering, frozen plans |
| `train.py` under `src/tool_rl/`; `configs/grpo.json` | TRL integration, four-sample advantages, audit logging, checkpointing |
| `report_results.py`, `reporting.py` | Regenerate retained numerical results |
| `scripts/export_bf16.py` | Export and verify a compact inference checkpoint on CPU |
| `configs/`, `repro/`, `results/`, `tests/` | Reported settings, provenance, compact evidence, CPU regression checks |

## Dependencies and credits

The model comes from [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Training reuses [Hugging Face TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer), Transformers, Accelerate, and PyTorch; it is not a newly implemented policy-gradient trainer. Symbolic grading uses [Math-Verify](https://github.com/huggingface/Math-Verify).

Data comes from [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), and [KodCode](https://huggingface.co/datasets/KodCode/KodCode-V1). Model and dataset use is subject to their respective upstream terms. Codex assisted with implementation, debugging, tests, and documentation.
