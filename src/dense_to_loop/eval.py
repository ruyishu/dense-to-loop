import json
import time
from pathlib import Path

import torch

from .data import prompt_ids, read_jsonl, write_jsonl
from .model import generate


def correct_answer(response, answer):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    extraction = (LatexExtractionConfig(), ExprExtractionConfig())
    # Restrict extraction to the final segment when a reasoning separator exists.
    response = response.rsplit("</think>", 1)[-1]
    gold = parse(f"${answer}$", extraction_config=extraction)
    prediction = parse(response, extraction_config=extraction)
    return bool(gold and prediction and verify(gold, prediction))


def evaluate(args, student, tokenizer, valid, eos):
    samples = read_jsonl(args.data)
    if args.limit is not None:
        samples = samples[:args.limit]
    if any(row.get("answer") in (None, "") for row in samples):
        raise ValueError("Each evaluation row requires a reference answer")
    student.eval()
    records = []
    before = time.perf_counter()
    for i, row in enumerate(samples):
        prefix = prompt_ids(tokenizer, row["prompt"])
        for j in range(args.samples):
            answer = generate(student, prefix, valid, eos, args.max_new, args.temperature, args.depth)
            response_ids = answer[:-1] if answer[-1] in eos else answer
            response = tokenizer.decode(response_ids, skip_special_tokens=False)
            records.append({"id": row.get("id", i), "sample": j, "correct": correct_answer(response, str(row["answer"])),
                            "prompt_ids": prefix, "answer_ids": answer,
                            "output_tokens": len(answer), "input_tokens": len(prefix),
                            "truncated": answer[-1] not in eos, "response": response})
    path = Path(args.output)
    write_jsonl(path, records)
    summary = {"questions": len(samples), "samples_per_question": args.samples, "depth": args.depth,
               "accuracy": sum(x["correct"] for x in records) / len(records),
               "mean_output_tokens": sum(x["output_tokens"] for x in records) / len(records),
               "truncation_rate": sum(x["truncated"] for x in records) / len(records),
               "wall_seconds_including_grading": time.perf_counter() - before}
    path.with_suffix(path.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def paired_report(one_path, two_path, seed):
    def by_question(path):
        grouped = {}
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["id"], row["sample"])
                if key in grouped:
                    raise ValueError(f"Duplicate evaluation key {key}")
                grouped[key] = bool(row["correct"])
        return grouped

    left, right = by_question(one_path), by_question(two_path)
    if left.keys() != right.keys() or not left:
        raise ValueError("Use the same ordered question set and sampling count")
    grouped = {}
    for (question, sample), value in left.items():
        grouped.setdefault(question, []).append(int(right[(question, sample)]) - int(value))
    deltas = torch.tensor([sum(x) / len(x) for x in grouped.values()])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(deltas), (2000, len(deltas)), generator=generator)
    confidence = deltas[indices].mean(dim=1).quantile(torch.tensor([0.025, 0.975])).tolist()
    print(json.dumps({"questions": len(deltas), "accuracy_delta_right_minus_left": deltas.mean().item(),
                      "paired_bootstrap_95": confidence,
                      "correct_to_wrong": sum(left[k] and not right[k] for k in left),
                      "wrong_to_correct": sum(not left[k] and right[k] for k in left)}, indent=2))
