import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class InputUpdater(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.embedding_norm = nn.RMSNorm(width, eps=eps)
        self.hidden_norm = nn.RMSNorm(width, eps=eps)
        self.fuse = nn.Linear(2 * width, width, bias=False)
        self.input_norm = nn.RMSNorm(width, eps=eps)
        self.gate = nn.Linear(width, width, bias=False)
        self.up = nn.Linear(width, width, bias=False)
        self.down = nn.Linear(width, width, bias=False)
        self.output_norm = nn.RMSNorm(width, eps=eps)

    def forward(self, embedding, hidden):
        combined = self.fuse(torch.cat((self.embedding_norm(embedding), self.hidden_norm(hidden)), dim=-1))
        x = self.input_norm(combined)
        return self.output_norm(self.down(F.silu(self.gate(x)) * self.up(x)))


def decoder_layer(layer, hidden, rotary, mask, preceding=None):
    attn = layer.self_attn
    norm = layer.input_layernorm(hidden)
    head_shape = (*norm.shape[:-1], -1, attn.head_dim)
    query = attn.q_norm(attn.q_proj(norm).view(head_shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(norm).view(head_shape)).transpose(1, 2)
    value = attn.v_proj(norm).view(head_shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *rotary)
    if preceding is not None:
        key = torch.cat((preceding[0], key), dim=-2)
        value = torch.cat((preceding[1], value), dim=-2)
    groups = query.size(1) // key.size(1)
    context = F.scaled_dot_product_attention(
        query, key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1),
        attn_mask=mask, scale=attn.scaling,
    )
    hidden = hidden + attn.o_proj(context.transpose(1, 2).reshape(*norm.shape[:-1], -1))
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, key, value


class FixedDepthQwen(nn.Module):
    """Shared Qwen3 layers with base-stream KV and current-token recurrence."""

    def __init__(self, backbone, loop, devices, recompute=True):
        super().__init__()
        self.backbone = backbone
        config = backbone.config
        if config.model_type != "qwen3" or getattr(config, "use_sliding_window", False):
            raise ValueError("Expected a full-attention Qwen3 checkpoint")
        self.recompute = recompute
        self.device = torch.device(devices[0])
        self.layer_devices = []
        model = backbone.model
        # Embedding and LM head may share a single parameter.
        model.embed_tokens.to(self.device)
        model.rotary_emb.to(self.device)
        model.norm.to(self.device)
        backbone.lm_head.to(self.device)
        for index, layer in enumerate(model.layers):
            device = torch.device(devices[index * len(devices) // len(model.layers)])
            self.layer_devices.append(device)
            layer.to(device)
        self.updater = None
        if loop:
            self.updater = InputUpdater(config.hidden_size, config.rms_norm_eps).to(
                device=self.device, dtype=model.embed_tokens.weight.dtype,
            )

    @property
    def head(self):
        return self.backbone.lm_head

    def states(self, ids, depth=None, past=None):
        depth = (2 if self.updater is not None else 1) if depth is None else depth
        if depth not in (1, 2) or (depth == 2 and self.updater is None):
            raise ValueError("This checkpoint does not support the requested depth")
        model = self.backbone.model
        embedding = model.embed_tokens(ids.to(self.device))
        length = ids.size(1)
        offset = 0 if past is None else past[0][0].size(-2)
        positions = torch.arange(offset, offset + length, device=self.device)
        rotary = model.rotary_emb(embedding, positions.unsqueeze(0))
        causal = torch.arange(offset + length, device=self.device)[None, :] <= positions[:, None]
        recurrent = torch.cat((causal, torch.eye(length, dtype=torch.bool, device=self.device)), dim=-1)
        tables = {d: (tuple(x.to(d) for x in rotary), causal.to(d), recurrent.to(d)) for d in set(self.layer_devices)}
        base = []
        hidden = embedding
        for index, (layer, device) in enumerate(zip(model.layers, self.layer_devices)):
            rope, mask, _ = tables[device]
            prefix = None if past is None else past[index]
            hidden, key, value = self._layer(layer, hidden.to(device), rope, mask, prefix)
            base.append((key, value))
        first = model.norm(hidden.to(self.device))
        if depth == 1:
            return (first,), base
        hidden = self.updater(embedding, first)
        for layer, device, prefix in zip(model.layers, self.layer_devices, base):
            rope, _, mask = tables[device]
            hidden, _, _ = self._layer(layer, hidden.to(device), rope, mask, prefix)
        return (first, model.norm(hidden.to(self.device))), base

    def _layer(self, layer, hidden, rotary, mask, prefix):
        if self.training and self.recompute and torch.is_grad_enabled():
            return checkpoint(decoder_layer, layer, hidden, rotary, mask, prefix, use_reentrant=False)
        return decoder_layer(layer, hidden, rotary, mask, prefix)


def load_student(name, revision, dtype, devices, loop):
    backbone = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
    return FixedDepthQwen(backbone, loop, devices)


def log_probabilities(head, states, valid):
    logits = [head(state).float().masked_fill(~valid, -torch.inf) for state in states]
    log_probs = [F.log_softmax(x, dim=-1) for x in logits]
    if len(log_probs) == 1:
        return log_probs[0]
    # Use finite operands on invalid vocabulary rows. logaddexp(-inf, -inf)
    # would otherwise produce undefined gradients even when the row is masked.
    first, second = (x.masked_fill(~valid, 0) for x in log_probs)
    return (torch.logaddexp(first, second) - math.log(2)).masked_fill(~valid, -torch.inf)


@torch.no_grad()
def generate(student, prompt, valid, eos, limit, temperature=1.0, depth=None):
    input_ids = torch.tensor([prompt], device=student.device)
    cache = None
    answer = []
    for _ in range(limit):
        states, cache = student.states(input_ids, depth=depth, past=cache)
        log_probs = log_probabilities(student.head, tuple(x[:, -1] for x in states), valid)
        if temperature == 0:
            input_ids = log_probs.argmax(dim=-1, keepdim=True)
        else:
            input_ids = torch.multinomial((log_probs / temperature).softmax(dim=-1), 1)
        token = input_ids.item()
        answer.append(token)
        if token in eos:
            break
    return answer
