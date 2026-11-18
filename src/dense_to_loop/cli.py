import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import model_info
from transformers import AutoTokenizer

from .data import prepare_data, prepare_math500
from .eval import evaluate, paired_report
from .model import load_student
from .train import atomic_save, loss_sum, train


DEFAULT_STUDENT = "Qwen/Qwen3-1.7B-Base"
DEFAULT_TEACHER = "Qwen/Qwen3-8B"
HARDWARE = {
    "3xl20": (["cuda:0", "cuda:1"], "cuda:2"),
    "2xl20": (["cuda:0"], "cuda:1"),
    "1xh20": (["cuda:0"], "cuda:0"),
    "cpu": (["cpu"], "cpu"),
}


def resolve_revision(name, revision):
    if Path(name).is_dir():
        return None
    if revision and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return revision
    return model_info(name, revision=revision).sha


def compatible_tokens(student, student_tokenizer, teacher_tokenizer):
    student_vocab = student_tokenizer.get_vocab()
    if student_vocab != teacher_tokenizer.get_vocab():
        raise ValueError("The teacher and student tokenizer vocabularies differ")
    size = student.head.weight.size(0)
    ids = sorted(set(student_vocab.values()))
    if not ids or ids[0] < 0 or ids[-1] >= size:
        raise ValueError("Tokenizer IDs exceed the LM head vocabulary")
    valid = torch.zeros(size, dtype=torch.bool, device=student.device)
    valid[ids] = True
    eos = {teacher_tokenizer.eos_token_id}
    if "<|im_end|>" in student_vocab:
        eos.add(student_vocab["<|im_end|>"])
    if "<|endoftext|>" in student_vocab:
        eos.add(student_vocab["<|endoftext|>"])
    eos.discard(None)
    return valid, eos


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def common(parser):
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--student-revision")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware", choices=HARDWARE, default="1xh20")
    parser.add_argument("--dense", action="store_true")


def arguments(argv):
    parser = argparse.ArgumentParser(prog="d2l")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--hardware", choices=HARDWARE, default="1xh20")
    data = commands.add_parser("prepare-data")
    data.add_argument("--output", default="data")
    data.add_argument("--train-count", type=positive, default=128)
    data.add_argument("--dev-count", type=positive, default=64)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--revision")
    data.add_argument("--math500", action="store_true")
    data.add_argument("--math500-revision")
    convert = commands.add_parser("convert")
    common(convert)
    convert.add_argument("--output", required=True)
    training = commands.add_parser("train")
    common(training)
    training.add_argument("--init")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--mode", choices=("opd", "sft", "offline-kd"), required=True)
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--budget", type=positive, required=True)
    training.add_argument("--steps", type=positive)
    training.add_argument("--tokens-per-step", type=positive, default=512)
    training.add_argument("--max-prompt", type=positive, default=1024)
    training.add_argument("--max-new", type=positive, default=256)
    training.add_argument("--chunk", type=positive, default=32)
    training.add_argument("--lr", type=float, default=1e-5)
    training.add_argument("--save-every", type=int, default=0)
    training.add_argument("--log-rollouts", action="store_true")
    training.add_argument("--updater-only", action="store_true")
    evaluation = commands.add_parser("eval")
    common(evaluation)
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--depth", type=int, choices=(1, 2))
    evaluation.add_argument("--max-new", type=positive, default=256)
    evaluation.add_argument("--samples", type=positive, default=1)
    evaluation.add_argument("--limit", type=positive)
    evaluation.add_argument("--temperature", type=float, default=0.0)
    probe = commands.add_parser("probe")
    common(probe)
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--trajectories", required=True)
    probe.add_argument("--limit", type=positive, default=32)
    probe.add_argument("--chunk", type=positive, default=32)
    report = commands.add_parser("report")
    report.add_argument("--left", required=True)
    report.add_argument("--right", required=True)
    report.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def check_hardware(name, teacher_needed=True):
    student, teacher = HARDWARE[name]
    all_devices = set(student) | ({teacher} if teacher_needed else set())
    gpu_ids = sorted(int(d.split(":")[1]) for d in all_devices if d.startswith("cuda:"))
    if gpu_ids and (not torch.cuda.is_available() or torch.cuda.device_count() <= gpu_ids[-1]):
        raise SystemExit(f"{name} requires {gpu_ids[-1] + 1} visible CUDA devices")
    return student, teacher, gpu_ids


def main(argv=None):
    args = arguments(argv)
    if args.command == "prepare-data":
        prepare_data(args.output, args.train_count, args.dev_count, args.seed, args.revision)
        if args.math500:
            prepare_math500(Path(args.output) / "math500.jsonl", args.math500_revision)
        return
    if args.command == "report":
        paired_report(args.left, args.right, args.seed)
        return
    devices, teacher_device, gpu_ids = check_hardware(args.hardware, args.command in ("doctor", "train", "probe"))
    if args.command == "doctor":
        results = [{"device": i, "name": torch.cuda.get_device_name(i),
                    "GiB": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)} for i in gpu_ids]
        print(json.dumps({"torch": torch.__version__, "devices": results, "student": devices, "teacher": teacher_device}))
        return

    torch.manual_seed(args.seed)
    checkpoint_path = None
    if args.command == "train":
        checkpoint_path = str(Path(args.output) / "last.pt") if args.resume else args.init
        if not checkpoint_path and not args.dense:
            raise SystemExit("Create a zero-step checkpoint with d2l convert and pass it as --init")
        if args.lr <= 0 or args.save_every < 0:
            raise SystemExit("Expected positive --lr and non-negative --save-every")
    elif args.command in ("eval", "probe"):
        checkpoint_path = args.checkpoint
        if args.command == "eval" and args.temperature < 0:
            raise SystemExit("Temperature must be non-negative")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True) if checkpoint_path else None
    config = payload["config"] if payload is not None else {}
    if payload is not None:
        args.student, args.student_revision = config["student"], config["student_revision"]
        args.teacher, args.teacher_revision = config["teacher"], config["teacher_revision"]
        args.dense = not config["loop"]
    if args.command == "train" and args.updater_only and args.dense:
        raise SystemExit("--updater-only requires a loop student")
    dtype = torch.float32 if args.hardware == "cpu" else torch.bfloat16
    student_revision = resolve_revision(args.student, args.student_revision)
    teacher_revision = resolve_revision(args.teacher, args.teacher_revision)
    metadata = {"student": args.student, "student_revision": student_revision, "teacher": args.teacher,
                "teacher_revision": teacher_revision, "loop": not args.dense, "dtype": str(dtype)}
    student = load_student(args.student, student_revision, dtype, devices, loop=not args.dense)
    if args.command == "convert":
        if args.dense:
            raise SystemExit("convert expects a loop student")
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        atomic_save({"config": metadata, "updater": student.updater.state_dict(), "step": 0}, args.output)
        print(json.dumps({"init": args.output, **metadata}))
        return
    if payload is not None:
        if args.command == "train" and not args.resume and (payload["step"] != 0 or "updater" not in payload):
            raise SystemExit("--init expects the zero-step output of convert")
        if "model" in payload:
            student.load_state_dict(payload["model"])
            payload.pop("model")
        else:
            student.updater.load_state_dict(payload["updater"])
    student_tokenizer = AutoTokenizer.from_pretrained(args.student, revision=student_revision)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, revision=teacher_revision)
    valid, eos = compatible_tokens(student, student_tokenizer, tokenizer)
    if args.command == "eval":
        args.depth = args.depth or (1 if args.dense else 2)
        evaluate(args, student, tokenizer, valid, eos)
        return
    teacher = load_student(args.teacher, teacher_revision, dtype, [teacher_device], loop=False)
    teacher.requires_grad_(False).eval()
    if teacher.head.weight.size(0) != valid.numel():
        raise ValueError("Teacher and student LM head vocabulary sizes differ")
    if args.command == "probe":
        student.eval()
        sums = {depth: 0.0 for depth in ((1,) if args.dense else (1, 2))}
        tokens = 0
        with torch.no_grad(), Path(args.trajectories).open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= args.limit:
                    break
                trace = json.loads(line)
                prompt, answer = trace["prompt_ids"], trace["answer_ids"]
                for depth in sums:
                    sums[depth] += loss_sum(student, teacher, prompt, answer, valid, "opd", args.chunk, depth).item()
                tokens += len(answer)
        if not tokens:
            raise ValueError("The probe needs at least one answer token")
        print(json.dumps({"tokens": tokens, "teacher_forward_kl_by_depth": {d: val / tokens for d, val in sums.items()}}))
        return
    train(args, student, teacher, tokenizer, valid, eos, metadata, payload)


if __name__ == "__main__":
    main()
