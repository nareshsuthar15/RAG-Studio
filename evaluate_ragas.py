"""
Offline RAGAS evaluator.
Input JSONL rows should contain: question, answer, contexts (list[str]), ground_truth.
Example:
{"question":"What is the policy?","answer":"The policy says ...","contexts":["Relevant source text"],"ground_truth":"Verified expected answer"}
Run: python evaluate_ragas.py evaluation_results.jsonl
"""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python evaluate_ragas.py <evaluation_results.jsonl>")
    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON on line {line_no}: {exc}")
        required = ("question", "answer", "contexts", "ground_truth")
        missing = [key for key in required if key not in item]
        if missing:
            raise SystemExit(f"Line {line_no} is missing fields: {', '.join(missing)}")
        if not isinstance(item["contexts"], list):
            raise SystemExit(f"Line {line_no}: contexts must be a list of strings.")
        rows.append(item)

    if not rows:
        raise SystemExit("No evaluation rows found.")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    except ImportError as exc:
        raise SystemExit("Install evaluation dependencies: pip install ragas datasets langchain-openai\n" + str(exc))

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print(result)
    output = path.with_name(path.stem + "_ragas_scores.json")
    output.write_text(json.dumps(result.to_pandas().to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"Saved score details to: {output}")

if __name__ == "__main__":
    main()
