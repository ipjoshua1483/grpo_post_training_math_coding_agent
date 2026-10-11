"""Minimal Qwen2.5 tokenizer checks for the explicit text protocol.

No response schema is installed: Tool RL parses the generated <tool_call> JSON
itself and never delegates tool selection or recovery to native function calling.
"""


def configure_tokenizer(tokenizer, model_type):
    """Add parsing metadata while preserving Qwen2.5's native template."""
    if model_type != "qwen2":
        raise ValueError("This run expects Qwen2.5-1.5B-Instruct (model_type=qwen2)")
    if not isinstance(tokenizer.chat_template, str):
        raise ValueError("Qwen2.5's chat template is required")
    if tokenizer.eos_token != "<|im_end|>":
        raise ValueError("Unexpected Qwen2.5 EOS token")
    tokenizer.response_schema = None
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
