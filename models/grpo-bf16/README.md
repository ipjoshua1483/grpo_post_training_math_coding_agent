---
library_name: transformers
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- grpo
- math
- code
---
# GRPO math and coding checkpoint

Qwen2.5-1.5B-Instruct after joint math/coding GRPO, provided as a complete BF16 inference checkpoint. All tensors were verified against the original weights converted to BF16; the package also loaded successfully as Qwen2ForCausalLM.

From the repository root, install the documented environment and prepare the fixed evaluation datasets/plans, then run:

```bash
git lfs install --local
git lfs pull --include="models/grpo-bf16/*.safetensors"
python infer.py --experiment E3 --gpus 8 --output runs/E3
```

Use `--gpus 1` for a single GPU, or `--domain math` / `--domain code` for one domain. E1 and E2 use the base model with their respective configurations. Optimizer and other training state are excluded. Base-model use remains subject to the upstream model's terms.

Upstream model: Qwen2.5-1.5B-Instruct by Alibaba Cloud, distributed under the included Apache 2.0 license. The weights were modified through joint GRPO post-training and exported to BF16.
