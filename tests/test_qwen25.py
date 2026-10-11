import unittest
from types import SimpleNamespace
from tool_rl.qwen25 import configure_tokenizer


class Qwen25Tests(unittest.TestCase):
    def tokenizer(self):
        return SimpleNamespace(chat_template="native <tool_call> template", eos_token="<|im_end|>", pad_token_id=151643)

    def test_adapter_preserves_template_and_disables_native_schema(self):
        tokenizer = self.tokenizer()
        original = tokenizer.chat_template
        self.assertIs(configure_tokenizer(tokenizer, "qwen2"), tokenizer)
        self.assertEqual(tokenizer.chat_template, original)
        self.assertIsNone(tokenizer.response_schema)
        self.assertEqual(tokenizer.padding_side, "left")

    def test_wrong_architecture_or_missing_chat_template_rejected(self):
        with self.assertRaises(ValueError):
            configure_tokenizer(self.tokenizer(), "qwen3")
        tokenizer = self.tokenizer()
        tokenizer.chat_template = None
        with self.assertRaises(ValueError):
            configure_tokenizer(tokenizer, "qwen2")


if __name__ == "__main__":
    unittest.main()
