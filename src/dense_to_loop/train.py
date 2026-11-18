import json
import os
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .data import dataset_hash, prompt_ids, read_jsonl
from .model import generate, log_probabilities


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def append_json(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class MasterAdamW:
    def __init__(self, parameters, lr):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.master = [torch.nn.Parameter(p.detach().float().clone()) if p.dtype != torch.float32 else p
                       for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.master, lr=lr, foreach=False)

    @torch.no_grad()
    def step(self, tokens):
        sq_norm = 0.0
        for param, master in zip(self.parameters, self.master):
            if param.grad is not None:
                master.grad = param.grad.float().div(tokens)
                sq_norm += master.grad.square().sum().item()
            if param is not master:
                param.grad = None
        norm = sq_norm ** 0.5
        if not torch.isfinite(torch.tensor(norm)):
            raise FloatingPointError("Non-finite gradient norm")
        scale = min(1.0, 1.0 / (norm + 1e-6))
        for master in self.master:
            if master.grad is not None:
                master.grad.mul_(scale)
        self.optimizer.step()
        for param, master in zip(self.parameters, self.master):
            if param is not master:
                param.copy_(master)
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def state_dict(self):
        return {"master": [p.detach() for p in self.master], "adam": self.optimizer.state_dict()}

    @torch.no_grad()
    def load_state_dict(self, state):
        for master, saved in zip(self.master, state["master"], strict=True):
            master.copy_(saved)
        self.optimizer.load_state_dict(state["adam"])


def loss_sum(student, teacher, prompt, answer, valid, mode, chunk, depth=None):
    ids = torch.tensor([prompt + answer], dtype=torch.long)
    left = len(prompt) - 1
    right = left + len(answer)
    states, _ = student.states(ids, depth=depth)
    states = tuple(x[0, left:right] for x in states)
    teacher_states = None
    if mode != "sft":
        with torch.no_grad():
            (hidden,), _ = teacher.states(ids, depth=1)
            teacher_states = hidden[0, left:right]

    def block(target_ids, target_states, *student_states):
        student_log = log_probabilities(student.head, student_states, valid)
        if target_states is None:
            return -student_log.gather(1, target_ids[:, None]).sum()
        with torch.no_grad():
            teacher_log = log_probabilities(teacher.head, (target_states,), valid.to(teacher.device))
            teacher_log = teacher_log.to(student.device)
        difference = teacher_log.masked_fill(~valid, 0) - student_log.masked_fill(~valid, 0)
        return (teacher_log.exp() * difference).sum()

    target = torch.tensor(answer, device=student.device)
    result = 0.0
    for i in range(0, len(answer), chunk):
        span = slice(i, i + chunk)
        reference = None if teacher_states is None else teacher_states[span]
        result = result + checkpoint(block, target[span], reference, *(x[span] for x in states), use_reentrant=False)
    return result


def train(args, student, teacher, tokenizer, valid, eos, metadata, initial):
    rows = read_jsonl(args.data)
    eligible = []
    for row in rows:
        tokens = prompt_ids(tokenizer, row["prompt"])
        if len(tokens) <= args.max_prompt:
            eligible.append((row, tokens))
    if not eligible:
        raise ValueError("All prompts exceed --max-prompt")
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    latest = run / "last.pt"
    if latest.exists() and not args.resume:
        raise FileExistsError(f"Use --resume or a new directory: {latest}")
    config = {**metadata, "data_sha256": dataset_hash(args.data), "tokens_per_step": args.tokens_per_step,
              "max_prompt": args.max_prompt, "max_new": args.max_new, "lr": args.lr,
              "mode": args.mode, "seed": args.seed, "updater_only": args.updater_only}
    if args.updater_only:
        student.backbone.requires_grad_(False)
    optimizer = MasterAdamW(student.parameters(), args.lr)
    step = position = tokens_seen = 0
    if args.resume:
        if initial["config"] != config:
            raise ValueError("Checkpoint configuration differs from this run")
        optimizer.load_state_dict(initial["optimizer"])
        restore_rng(initial["rng"])
        step, position, tokens_seen = (initial[k] for k in ("step", "position", "tokens"))
        for filename, field, maximum in (("metrics.jsonl", "step", step), ("rollouts.jsonl", "cursor", position)):
            path = run / filename
            if path.exists():
                tmp = path.with_suffix(".tmp")
                with path.open(encoding="utf-8") as source, tmp.open("w", encoding="utf-8") as dest:
                    for line in source:
                        try:
                            saved = json.loads(line)
                        except json.JSONDecodeError:
                            break  # Interrupted append at the end of the file.
                        if saved[field] <= maximum:
                            dest.write(line)
                os.replace(tmp, path)
    start_step = step
    start_tokens = tokens_seen
    start_time = time.perf_counter()
    devices = set(student.layer_devices) | {teacher.device, student.device}
    for device in devices:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def save():
        atomic_save({"config": config, "model": student.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng": rng_state(), "step": step, "position": position, "tokens": tokens_seen}, latest)

    while tokens_seen < args.budget and (args.steps is None or step - start_step < args.steps):
        total_loss = 0.0
        tokens_in_step = 0
        sample_time = train_time = 0.0
        loop_tokens = teacher_forward_tokens = teacher_generated = truncated = trajectories = 0
        student_generation_tokens = teacher_generation_tokens = 0
        updater_gradient = 0.0
        target = min(args.tokens_per_step, args.budget - tokens_seen)
        while tokens_in_step < target:
            index = position % len(eligible)
            row, prompt = eligible[index]
            position += 1
            trajectories += 1
            limit = min(args.max_new, target - tokens_in_step)
            before = time.perf_counter()
            student.eval()
            if args.mode == "opd":
                answer = generate(student, prompt, valid, eos, limit)
                student_generation_tokens += len(prompt) + len(answer) - 1
            elif "response" in row:
                answer = tokenizer(row["response"], add_special_tokens=False)["input_ids"][:limit]
                if not answer:
                    raise ValueError(f"Empty teacher response at row {index}")
            else:
                answer = generate(teacher, prompt, valid.to(teacher.device), eos, limit)
                teacher_generated += len(answer)
                teacher_generation_tokens += len(prompt) + len(answer) - 1
            truncated += int(answer[-1] not in eos and len(answer) == limit)
            sample_time += time.perf_counter() - before
            before = time.perf_counter()
            student.train()
            loss = loss_sum(student, teacher, prompt, answer, valid, args.mode, args.chunk)
            loss.backward()
            total_loss += loss.detach().float().item()
            train_time += time.perf_counter() - before
            loop_tokens += len(prompt) + len(answer)
            teacher_forward_tokens += (len(prompt) + len(answer)) if args.mode != "sft" else 0
            tokens_in_step += len(answer)
            if args.log_rollouts:
                append_json(run / "rollouts.jsonl", {"policy_step": step, "cursor": position,
                            "id": row.get("id", index), "prompt_ids": prompt, "answer_ids": answer,
                            "stop": "eos" if answer[-1] in eos else "length" if len(answer) == limit else "reference_end"})
        if student.updater is not None:
            updater_gradient = sum(p.grad.detach().float().norm().item() for p in student.updater.parameters() if p.grad is not None)
        before = time.perf_counter()
        grad_norm = optimizer.step(tokens_in_step)
        if student.device.type == "cuda":
            for device in set(student.layer_devices):
                torch.cuda.synchronize(device)
        update_time = time.perf_counter() - before
        step += 1
        tokens_seen += tokens_in_step
        record = {"step": step, "tokens": tokens_seen, "answer_tokens": tokens_in_step,
                  "trajectories": trajectories, "truncated": truncated, "loss": total_loss / tokens_in_step,
                  "grad_norm": grad_norm, "updater_grad_norm_sum": updater_gradient,
                  "student_replay_tokens": loop_tokens, "teacher_scoring_tokens": teacher_forward_tokens,
                  "student_generation_forward_tokens": student_generation_tokens,
                  "teacher_generation_forward_tokens": teacher_generation_tokens,
                  "teacher_generated_tokens": teacher_generated, "sample_seconds": sample_time,
                  "replay_seconds": train_time, "update_seconds": update_time}
        append_json(run / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        if args.save_every and step % args.save_every == 0:
            save()
    save()
    elapsed = time.perf_counter() - start_time
    summary = {"steps": step - start_step, "new_answer_tokens": tokens_seen - start_tokens,
               "wall_seconds_including_checkpoints": elapsed,
               "answer_tokens_per_second": (tokens_seen - start_tokens) / elapsed,
               "peak_allocated_gib": {str(d): torch.cuda.max_memory_allocated(d) / 2**30
                                      for d in devices if d.type == "cuda"}}
    (run / "last_session.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
