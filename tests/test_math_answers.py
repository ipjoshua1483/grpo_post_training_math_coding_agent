import unittest
from tool_rl.math_answers import extract_math_answer
from tool_rl.protocol import make_prompt
from tool_rl.rewards import grade


class MathAnswerTests(unittest.TestCase):
    def test_reported_tool_trajectory_extracts_correct_untagged_answer(self):
        messages = [
            {"role": "assistant", "content": "Calculating", "tool_calls": [{"name": "execute_python"}]},
            {"role": "tool", "content": '{"stdout":"29\\n"}'},
            {"role": "assistant", "content": "The calculation shows that Tom's profit last month was $29.\n\nTherefore, the final answer is: 29"},
        ]
        result = grade({"domain": "math", "answer": "29"}, messages)
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["extracted_answer"], "29")
        self.assertFalse(result["format_ok"])
        self.assertIn("answer_cue", result["extraction_method"])

    def test_numeric_wording_currency_and_units(self):
        for response in ("The answer is 29.", "Final answer: $29", "<final>The answer is 29 dollars.</final>", "29", r"Therefore, the answer is \(29\).", r"Work first. \boxed{29}", "Tom's profit last month was $29."):
            with self.subTest(response=response):
                self.assertEqual(grade({"domain": "math", "answer": "29"}, response)["reward"], 1)

    def test_wrong_conclusion_never_matches_an_intermediate_gold_number(self):
        for response in ("Intermediate result: 29. The final answer is 30", "29 or 30", "<final>29 or 30</final>", "29 is wrong", "I computed 29 but this is incorrect.", "<final>29</final><final>30</final>"):
            with self.subTest(response=response):
                self.assertEqual(grade({"domain": "math", "answer": "29"}, response)["reward"], 0)

    def test_no_reward_for_tool_output_or_unfinished_submissions(self):
        problem = {"domain": "math", "answer": "29"}
        self.assertEqual(grade(problem, [{"role": "tool", "content": "29"}])["reward"], 0)
        self.assertEqual(grade(problem, [{"role": "assistant", "content": "29", "tool_calls": [{}]}])["reward"], 0)
        self.assertEqual(grade(problem, "The answer is 29", truncated=True)["reward"], 0)
        self.assertEqual(grade(problem, "<final>29")["reward"], 0)

    def test_symbolic_cue_extracts_expression_without_using_gold(self):
        extracted = extract_math_answer(r"Some explanation. Therefore, the final answer is: \(2\sqrt{2}\).")
        self.assertEqual(extracted["answer"], r"2\sqrt{2}")

    def test_no_tool_numeric_baseline_uses_the_same_extractor(self):
        public = {"domain": "math", "id": "dev/0", "benchmark": "gsm8k", "prompt": "Compute"}
        ref = {"domain": "math", "answer": "29", "verifier": "numeric"}
        result = grade(ref, "The final answer is: 29", truncated=False)
        self.assertTrue(result["correct"])
        self.assertFalse(result["format_ok"])

    def test_math_prompt_is_flexible_and_domain_specific(self):
        problem = {"domain": "math", "prompt": "Compute"}
        for prompt in (make_prompt(problem), make_prompt(problem, tools_enabled=False)):
            self.assertIn("You may explain your reasoning", prompt[0]["content"])
            self.assertIn("Prefer marking it", prompt[0]["content"])
            self.assertNotIn("MANDATORY", prompt[0]["content"])
            self.assertNotIn("Return ONLY", prompt[-1]["content"])
            self.assertNotIn("For coding", prompt[0]["content"])

    def test_extraction_priority_prefers_tags_then_boxes_then_cues(self):
        self.assertEqual(extract_math_answer(r"\boxed{30} The answer is 31. <final>29</final>")["answer"], "29")
        self.assertEqual(extract_math_answer(r"The answer is 30. \boxed{29}")["answer"], "29")
        self.assertEqual(extract_math_answer("The answer is 29\n30")["answer"], "29")

    def test_last_line_fallback_is_mathematical_not_arbitrary_prose(self):
        for text, answer in (("Reasoning about the problem.\n29", "29"), (r"Reasoning." + "\n" + r"\frac{1}{2}", r"\frac{1}{2}"), ("Work first.\nx^2+2x+1", "x^2+2x+1")):
            result = extract_math_answer(text)
            self.assertEqual(result["answer"], answer)
            self.assertEqual(result["method"], "final_sentence")
        for text in ("Work: 29\nI am unsure", "29\n30", "29 or 30", r"\boxed{29} or \boxed{30}", "The final answer is 29\nThe final answer is 30"):
            self.assertIsNone(extract_math_answer(text)["answer"])

    def test_observed_smoke_final_sentences(self):
        examples = (
            ("Therefore, the candy bar cost **131** cents.", "131"),
            (r"Therefore, the value of \( f(g(2)) \) is \( 1 \).", "1"),
            ("The calculation shows that you can make 30 different shirt-and-tie outfits.", "30"),
        )
        for response, expected in examples:
            with self.subTest(response=response):
                result = extract_math_answer(response)
                self.assertEqual(result["answer"], expected)
                self.assertEqual(result["method"], "final_sentence")

    def test_final_sentence_fallback_rejects_multiple_candidates(self):
        for response in (
            "It could be 29 or 31.",
            "Therefore the possibilities are **29** and **31**.",
            r"The result is either \(29\) or \(31\).",
        ):
            with self.subTest(response=response):
                result = extract_math_answer(response)
                self.assertIsNone(result["answer"])
                self.assertIn("ambiguous", result["method"])

    def test_fallback_does_not_search_prior_reasoning(self):
        result = extract_math_answer("An intermediate computation gives 131.\nI cannot determine the answer.")
        self.assertIsNone(result["answer"])

    def test_smoke_semantic_outcomes_remain_binary(self):
        cases = (
            ("Therefore, the candy bar cost **131** cents.", "131", 1.0),
            (r"Therefore, the value of \( f(g(2)) \) is \( 1 \).", "1", 1.0),
            ("The calculation shows that you can make 30 different shirt-and-tie outfits.", "30", 1.0),
            ("Therefore, Davey has 9 guitars.", "18", 0.0),
            (r"Thus, \(x = 3^{2/9}\).", "9", 0.0),
        )
        for response, gold, reward in cases:
            with self.subTest(response=response):
                result = grade({"domain": "math", "answer": gold}, response)
                self.assertEqual(result["reward"], reward)
                self.assertFalse(result["format_ok"])

    def test_reasoning_does_not_break_optional_final_tag_metric(self):
        result = grade({"domain": "math", "answer": "29"}, "Reasoning first.\n<final>29</final>")
        self.assertTrue(result["format_ok"])
        self.assertEqual(result["reward"], 1)

    def test_no_tool_prompt_does_not_suggest_calling_tools(self):
        prompt = make_prompt({"domain": "math", "prompt": "Compute"}, tools_enabled=False)
        self.assertIn("No execution tools", prompt[0]["content"])
        self.assertNotIn("execute_python", prompt[0]["content"])


if __name__ == "__main__":
    unittest.main()
