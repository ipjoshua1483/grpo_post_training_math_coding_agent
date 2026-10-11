"""Check numeric/symbolic grading in real containers using synthetic and dev answers.

No model inference, training, or held-out model evaluation occurs here.
"""
import argparse
from .experiment_data import load_split, select_rows
from .io import write_json
from .rewards import grade
from .sandbox import DockerSandbox


def run_checks(sandbox, data_dir=None):
    metadata = sandbox.preflight()
    probes = [
        ("numeric_equivalent", "numeric", "1000", "1,000", True),
        ("numeric_wrong", "numeric", "1000", "1001", False),
        ("numeric_latex_exact", "numeric", "0.5", r"\frac{1}{2}", True),
        ("numeric_latex_wrong", "numeric", "0.5", r"\frac{2}{3}", False),
        ("numeric_latex_near_miss", "numeric", "0.5", r"\frac{500001}{1000000}", False),
        ("numeric_untagged_wording", "numeric", "29", "Therefore, the final answer is: 29", True),
        ("numeric_ambiguous", "numeric", "29", "29 or 30", False),
        ("numeric_negated", "numeric", "29", "29 is wrong", False),
        ("symbolic_fraction", "symbolic", r"\frac{1}{2}", "0.5", True),
        ("symbolic_radical", "symbolic", r"\sqrt{8}", r"2\sqrt{2}", True),
        ("symbolic_answer_wording", "symbolic", r"\sqrt{8}", r"The final answer is: \(2\sqrt{2}\).", True),
        ("symbolic_polynomial", "symbolic", "x^2+2x+1", "(x+1)^2", True),
        ("symbolic_tuple", "symbolic", "(1,2)", "(1,2)", True),
        ("symbolic_set", "symbolic", r"\{1,2\}", r"\{2,1\}", True),
        ("symbolic_wrong", "symbolic", r"\frac{1}{2}", r"\frac{2}{3}", False),
        ("symbolic_invalid", "symbolic", "42", "not a mathematical answer", False),
    ]
    if data_dir is not None:
        rows, refs, _ = load_split(data_dir, "dev")
        for row in select_rows(rows, {"math": 4}, 42, "verifier-development"):
            ref = refs[row["id"]]
            probes.append(("development/" + row["id"], "symbolic", ref["answer"], ref["answer"], True))
    checks = []
    for name, verifier, gold, answer, expected in probes:
        result = grade({"domain": "math", "verifier": verifier, "answer": gold}, f"<final>{answer}</final>", sandbox)
        healthy = result.get("verification", {}).get("ok", True) is True
        passed = healthy and result["correct"] == expected
        checks.append({"name": name, "passed": passed, "expected_correct": expected, "score": result})
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
    return {"kind": "verifier_integration_checks_not_model_results", "sandbox": metadata, "passed": all(row["passed"] for row in checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/frozen", help="Use math development references only")
    parser.add_argument("--output", default="runs/math-verifier-checks.json")
    args = parser.parse_args()
    result = run_checks(DockerSandbox(), args.data_dir)
    write_json(args.output, result)
    if not result["passed"]:
        raise SystemExit("Math verifier checks failed; inspect the report before evaluation")


if __name__ == "__main__":
    main()
