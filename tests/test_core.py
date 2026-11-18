import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from dense_to_loop.cli import main
from dense_to_loop.model import FixedDepthQwen, log_probabilities
from dense_to_loop.train import loss_sum


def tiny(width):
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=width, intermediate_size=2 * width,
                                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                       head_dim=width // 2, max_position_embeddings=32, tie_word_embeddings=True))


class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_first_pass_equals_dense_and_cached_second_pass_equals_full(self):
        base = tiny(16).eval()
        loop = FixedDepthQwen(base, True, ["cpu"]).eval()
        ids = torch.tensor([[1, 4, 5, 6]])
        with torch.no_grad():
            (first, second), _ = loop.states(ids)
            expected = base(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(loop.head(first), expected)
            _, cache = loop.states(ids[:, :-1])
            cached, _ = loop.states(ids[:, -1:], past=cache)
            torch.testing.assert_close(first[:, -1:], cached[0])
            torch.testing.assert_close(second[:, -1:], cached[1])
            changed, _ = loop.states(torch.tensor([[1, 4, 9, 8]]))
            torch.testing.assert_close(second[:, :2], changed[1][:, :2])

    def test_chunked_kl_matches_full_kl_and_trains_updater(self):
        loop = FixedDepthQwen(tiny(16), True, ["cpu"])
        teacher = FixedDepthQwen(tiny(24), False, ["cpu"]).requires_grad_(False).eval()
        valid = torch.tensor([True] * 14 + [False] * 2)
        prompt, answer = [1, 4, 3], [7, 2, 5]
        full = loss_sum(loop, teacher, prompt, answer, valid, "opd", 3)
        full.backward()
        expected = loop.updater.fuse.weight.grad.detach().clone()
        loop.zero_grad(set_to_none=True)
        chunked = loss_sum(loop, teacher, prompt, answer, valid, "opd", 2)
        chunked.backward()
        torch.testing.assert_close(full, chunked)
        torch.testing.assert_close(loop.updater.fuse.weight.grad, expected)
        self.assertGreater(expected.norm().item(), 0)
        self.assertTrue(torch.isfinite(expected).all())
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        states, _ = loop.states(torch.tensor([[1, 4]]))
        mixed = log_probabilities(loop.head, states, valid).exp()
        torch.testing.assert_close(mixed.sum(-1), torch.ones_like(mixed[..., 0]))

    def test_cli_conversion_online_steps_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student, teacher = root / "student", root / "teacher"
            vocab = {"[UNK]": 0, "[END]": 1, "What": 2, "is": 3, "2": 4, "?": 5, "4": 6}
            backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[END]")
            tokenizer.chat_template = "{% for item in messages %}{{ item.content }}{% endfor %}"
            for width, path in [(16, student), (24, teacher)]:
                tiny(width).save_pretrained(path)
                tokenizer.save_pretrained(path)
            data = root / "math.jsonl"
            data.write_text('{"id": "1", "prompt": "What is 2+2?", "answer": "4", "response": "4"}\n')
            init, run = root / "init.pt", root / "opd"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["convert", "--hardware", "cpu", "--student", str(student), "--teacher", str(teacher), "--output", str(init)])
                args = ["train", "--hardware", "cpu", "--mode", "opd", "--data", str(data), "--output", str(run),
                        "--tokens-per-step", "3", "--max-new", "2", "--chunk", "1", "--log-rollouts"]
                main(args + ["--budget", "3", "--init", str(init)])
                main(args + ["--budget", "6", "--resume"])
                for mode in ("sft", "offline-kd"):
                    main(args + ["--mode", mode, "--output", str(root / mode), "--budget", "3", "--init", str(init)])
            start = torch.load(init, weights_only=True)
            end = torch.load(run / "last.pt", weights_only=True)
            self.assertEqual((end["tokens"], end["step"]), (6, 2))
            self.assertFalse(torch.equal(start["updater"]["fuse.weight"], end["model"]["updater.fuse.weight"]))
            log = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(x["updater_grad_norm_sum"] > 0 for x in log))
            for mode in ("sft", "offline-kd"):
                self.assertTrue((root / mode / "last.pt").is_file())


if __name__ == "__main__":
    unittest.main()
