import hashlib
import json
import os
import random
from pathlib import Path


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(x.get("prompt"), str) or not x["prompt"].strip() for x in rows):
        raise ValueError(f"Expected nonempty JSONL rows with a string prompt in {path}")
    return rows


def prepare_data(directory, train_count, dev_count, seed, revision):
    from datasets import load_dataset

    if min(train_count, dev_count) < 1:
        raise ValueError("Train and dev counts must both be positive")
    ds = load_dataset("openai/gsm8k", "main", revision=revision)
    # Development and test questions have disjoint source splits. Remove exact
    # normalized duplicates before drawing the pilot training subset.
    seen = set()

    def rows(split):
        for i, row in enumerate(ds[split]):
            question = " ".join(row["question"].split()).casefold()
            if question in seen:
                continue
            seen.add(question)
            yield {"id": f"gsm8k-{split}-{i}", "prompt": row["question"], "answer": row["answer"].split("####")[-1].strip()}

    test = list(rows("test"))
    train = list(rows("train"))
    random.Random(seed).shuffle(train)
    if train_count + dev_count > len(train):
        raise ValueError("Too many training and development rows requested")
    directory = Path(directory)
    write_jsonl(directory / "train.jsonl", train[:train_count])
    write_jsonl(directory / "dev.jsonl", train[train_count:train_count + dev_count])
    write_jsonl(directory / "gsm8k.jsonl", test)


def prepare_math500(path, revision):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=revision)
    write_jsonl(path, ({"id": f"math500-{i}", "prompt": row["problem"], "answer": row["answer"]}
                       for i, row in enumerate(ds)))


def dataset_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_ids(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt + "\nPut your final answer in \\boxed{} ."}]
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
        return_dict=False,
    )
    return list(result)
