"""Qwen2.5 loading and token-preserving text/native generation adapters."""
import hashlib
import json
import os
from pathlib import Path
from .qwen25 import configure_tokenizer


def parse_qwen_native_response(text):
    """Parse Qwen2.5's textual tool syntax without tokenizer response_schema.

    Qwen2.5 renders native tools as ``<tool_call>`` JSON in ordinary text.
    This project deliberately leaves ``tokenizer.response_schema`` unset, so
    TRL's ``parse_response`` cannot be used.  Invalid calls are represented as
    structured invalid calls so the rollout controller can return corrective
    feedback instead of crashing or treating them as final answers.
    """
    from .protocol import parse_tool_action

    action = parse_tool_action(text)
    message = {"role": "assistant", "content": text}
    if action["kind"] == "final":
        return message
    if action["kind"] == "tool_call":
        message["tool_calls"] = [{
            "type": "function",
            "function": {
                "name": action["name"],
                "arguments": {"code": action["code"]},
            },
        }]
        return message

    # Keep malformed/wrong-name requests visible to the controller.  The
    # deliberately incomplete shape fails valid_call() and consumes only the
    # invalid-call budget, never the execute budget.
    message["tool_calls"] = [{
        "type": "invalid",
        "function": {
            "name": action.get("requested_name", "unknown"),
            "arguments": {},
        },
        "parse_error": action["error"],
    }]
    return message


def load_runtime(model_id, *, revision=None, seed=42, training=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl.chat_template_utils import is_chat_template_prefix_preserving
    if not torch.cuda.is_available():
        raise RuntimeError("GPU runtime pending: CUDA is unavailable")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid WORLD_SIZE/RANK environment")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError("LOCAL_RANK does not identify a visible CUDA device")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    local = Path(model_id).is_dir()
    resolved = revision if local else (revision or HfApi().model_info(model_id).sha)
    architecture = AutoConfig.from_pretrained(model_id, revision=resolved, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved, trust_remote_code=False)
    tokenizer = configure_tokenizer(tokenizer, architecture.model_type)
    if not is_chat_template_prefix_preserving(tokenizer):
        raise ValueError("Qwen2.5's chat template must preserve generation prefixes")
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=resolved, dtype=torch.float32 if training else torch.bfloat16, trust_remote_code=False).to(device)
    if not training:
        model.eval()
    origin = {"model": model_id, "revision": resolved}
    if local and (Path(model_id) / "tool_rl_training.json").exists():
        origin = json.loads((Path(model_id) / "tool_rl_training.json").read_text())["origin"]
    info = {"model": model_id, "revision": resolved, "origin": origin, "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(), "gpu": torch.cuda.get_device_name(local_rank), "cuda": torch.version.cuda, "parameter_dtype": str(next(model.parameters()).dtype), "rank": rank, "local_rank": local_rank, "world_size": world_size, "device": str(device)}
    return model, tokenizer, info


class HuggingFaceTurns:
    def __init__(self, model, tokenizer, policy):
        self.model, self.tokenizer, self.policy = model, tokenizer, policy

    def start(self, prompt, tools):
        if tools is not None:
            raise ValueError("Native tool schemas are disabled; use the textual controller")
        self.ids = list(self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=self.policy["enable_thinking"]))
        self.prompt_length = len(self.ids)
        self.prompt_ids = list(self.ids)
        self.completion_mask = []
        if self.prompt_length > self.policy["max_prompt_tokens"]:
            raise ValueError("Prompt exceeds configured budget; do not silently truncate problems")

    def next_turn(self):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopAtProtocolBoundary(StoppingCriteria):
            """Return control as soon as a textual tool/final block is complete."""

            def __init__(self, prompt_length, marker_sequences):
                self.prompt_length = prompt_length
                self.marker_sequences = marker_sequences

            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids[0, self.prompt_length:].tolist()
                matched = any(
                    len(generated) >= len(marker)
                    and generated[-len(marker):] == marker
                    for marker in self.marker_sequences
                )
                return torch.tensor([matched], dtype=torch.bool, device=input_ids.device)
        remaining = self.policy["max_completion_length"] - (len(self.ids) - self.prompt_length)
        context = self.model.config.max_position_embeddings - len(self.ids)
        if min(remaining, context) <= 0:
            return {"error": "token_budget" if remaining <= 0 else "context_budget", "generated_tokens": 0}
        device = next(self.model.parameters()).device
        ids = torch.tensor([self.ids], device=device)
        kwargs = {"max_new_tokens": min(remaining, context), "do_sample": self.policy["temperature"] > 0, "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id, "disable_compile": True}
        marker_sequences = [
            self.tokenizer.encode(marker, add_special_tokens=False)
            for marker in ("</tool_call>", "</final>")
        ]
        marker_sequences = [sequence for sequence in marker_sequences if sequence]
        kwargs["stopping_criteria"] = StoppingCriteriaList([
            StopAtProtocolBoundary(len(self.ids), marker_sequences)
        ])
        if self.policy["temperature"] > 0:
            kwargs.update(temperature=self.policy["temperature"], top_p=self.policy["top_p"])
        with torch.inference_mode():
            result = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
        tokens = result[0, len(self.ids):].tolist()
        self.ids.extend(tokens)
        self.completion_mask.extend([1] * len(tokens))
        if not tokens:
            return {"error": "token_budget", "generated_tokens": 0}
        text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        stopped_at_marker = any(
            len(tokens) >= len(marker) and tokens[-len(marker):] == marker
            for marker in marker_sequences
        )
        return {"message": {"role": "assistant", "content": text}, "generated_tokens": len(tokens), "terminated": tokens[-1] == self.tokenizer.eos_token_id or stopped_at_marker}

    def observe(self, messages):
        # Render textual observations as ordinary user turns. Generated assistant
        # IDs are already present, so append only the observation and next-turn
        # generation suffix. These external tokens receive a zero policy mask.
        dummy = [{"role": "user", "content": "dummy"}, {"role": "assistant", "content": "dummy"}]
        prefix = self.tokenizer.apply_chat_template(dummy, tokenize=True, add_generation_prompt=False, return_dict=False, enable_thinking=self.policy["enable_thinking"])
        full = self.tokenizer.apply_chat_template(dummy + messages, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=self.policy["enable_thinking"])
        if full[:len(prefix)] != prefix:
            raise RuntimeError("Chat template is not prefix preserving")
        suffix = list(full[len(prefix):])
        self.ids.extend(suffix)
        self.completion_mask.extend([0] * len(suffix))

    @property
    def completion_ids(self):
        return self.ids[self.prompt_length:]


class NativeToolHuggingFaceTurns:
    """Historical Qwen native-tool rendering inside the current evaluator."""

    def __init__(self, model, tokenizer, policy):
        self.model, self.tokenizer, self.policy = model, tokenizer, policy

    def start(self, prompt, tools):
        if not tools:
            raise ValueError("The compact native condition requires execute_python")
        self.ids = list(self.tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=self.policy["enable_thinking"],
        ))
        self.prompt_length = len(self.ids)
        self.prompt_ids = list(self.ids)
        self.completion_mask = []
        if self.prompt_length > self.policy["max_prompt_tokens"]:
            raise ValueError("Prompt exceeds configured budget; do not silently truncate problems")

    def next_turn(self):
        import torch

        remaining = self.policy["max_completion_length"] - (len(self.ids) - self.prompt_length)
        context = self.model.config.max_position_embeddings - len(self.ids)
        if min(remaining, context) <= 0:
            return {
                "error": "token_budget" if remaining <= 0 else "context_budget",
                "generated_tokens": 0,
            }
        device = next(self.model.parameters()).device
        ids = torch.tensor([self.ids], device=device)
        kwargs = {
            "max_new_tokens": min(remaining, context),
            "do_sample": self.policy["temperature"] > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "disable_compile": True,
        }
        if self.policy["temperature"] > 0:
            kwargs.update(
                temperature=self.policy["temperature"], top_p=self.policy["top_p"]
            )
        with torch.inference_mode():
            result = self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs
            )
        tokens = result[0, len(self.ids):].tolist()
        self.ids.extend(tokens)
        self.completion_mask.extend([1] * len(tokens))
        if not tokens:
            return {"error": "token_budget", "generated_tokens": 0}
        text = self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return {
            "message": parse_qwen_native_response(text),
            "generated_tokens": len(tokens),
            "terminated": tokens[-1] == self.tokenizer.eos_token_id,
        }

    def observe(self, messages):
        dummy = [
            {"role": "user", "content": "dummy"},
            {"role": "assistant", "content": "dummy"},
        ]
        prefix = self.tokenizer.apply_chat_template(
            dummy,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            enable_thinking=self.policy["enable_thinking"],
        )
        full = self.tokenizer.apply_chat_template(
            dummy + messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=self.policy["enable_thinking"],
        )
        if full[:len(prefix)] != prefix:
            raise RuntimeError("Chat template is not prefix preserving")
        suffix = list(full[len(prefix):])
        self.ids.extend(suffix)
        self.completion_mask.extend([0] * len(suffix))

    @property
    def completion_ids(self):
        return self.ids[self.prompt_length:]
