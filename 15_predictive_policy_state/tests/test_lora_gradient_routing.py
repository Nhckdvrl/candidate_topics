"""The capacity-restored G1 arm needs LoRA to receive action-loss gradient ONLY.

The trainer implements this by backpropagating the action term first, snapshotting the LoRA
grads, backpropagating the video term, then restoring the snapshot. This test verifies that
algorithm on a toy graph in which the "backbone" is genuinely shared by both objectives, so a
naive combined backward would contaminate the LoRA grads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4, bias=False)   # stands in for the WAM adapters
        self.lora_q = nn.Linear(4, 4, bias=False)   # action-only capacity
        self.video_head = nn.Linear(4, 1, bias=False)
        self.action_head = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        h = self.shared(x) + self.lora_q(x)
        return self.video_head(h).sum(), self.action_head(h).sum()


def _lora_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and ".lora_" in f".{n}"]


def _routed_backward(model, video_loss, action_loss):
    """Mirror of Wan22Trainer._backward_with_lora_action_only."""
    lora = _lora_params(model)
    action_loss.backward(retain_graph=True)
    snapshot = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in lora}
    video_loss.backward()
    for n, p in lora:
        saved = snapshot[n]
        if saved is None:
            if p.grad is not None:
                p.grad.zero_()
        else:
            p.grad.copy_(saved)


def test_lora_receives_action_gradient_only():
    torch.manual_seed(0)
    x = torch.randn(3, 4)

    model = _Toy()
    v, a = model(x)
    model.zero_grad(set_to_none=True)
    _routed_backward(model, v, a)
    routed = {
        n: (torch.zeros_like(p) if p.grad is None else p.grad.clone())
        for n, p in model.named_parameters()
    }

    # reference 1: action loss alone -> what LoRA must equal
    ref = _Toy()
    ref.load_state_dict(model.state_dict())
    _, a_ref = ref(x)
    ref.zero_grad(set_to_none=True)
    a_ref.backward()
    action_only = {
        n: (torch.zeros_like(p) if p.grad is None else p.grad.clone())
        for n, p in ref.named_parameters()
    }

    # reference 2: combined backward -> what the shared module must equal
    both = _Toy()
    both.load_state_dict(model.state_dict())
    v_b, a_b = both(x)
    both.zero_grad(set_to_none=True)
    (v_b + a_b).backward()
    combined = {
        n: (torch.zeros_like(p) if p.grad is None else p.grad.clone())
        for n, p in both.named_parameters()
    }

    # LoRA sees action gradient only ...
    assert torch.allclose(routed["lora_q.weight"], action_only["lora_q.weight"], atol=1e-6)
    # ... and is genuinely different from the contaminated combined gradient,
    # otherwise the test would pass trivially.
    assert not torch.allclose(routed["lora_q.weight"], combined["lora_q.weight"], atol=1e-6)

    # the shared module still receives BOTH objectives, unchanged.
    assert torch.allclose(routed["shared.weight"], combined["shared.weight"], atol=1e-6)
    assert not torch.allclose(routed["shared.weight"], action_only["shared.weight"], atol=1e-6)

    # future-only and action-only heads are untouched by the routing.
    assert torch.allclose(routed["video_head.weight"], combined["video_head.weight"], atol=1e-6)
    assert torch.allclose(routed["action_head.weight"], combined["action_head.weight"], atol=1e-6)


def test_routing_is_a_noop_when_video_term_is_zero():
    """future-off arm: the same code path must run, and must change nothing."""
    torch.manual_seed(1)
    x = torch.randn(3, 4)

    model = _Toy()
    v, a = model(x)
    model.zero_grad(set_to_none=True)
    _routed_backward(model, 0.0 * v, a)
    routed = {
        n: (torch.zeros_like(p) if p.grad is None else p.grad.clone())
        for n, p in model.named_parameters()
    }

    ref = _Toy()
    ref.load_state_dict(model.state_dict())
    _, a_ref = ref(x)
    ref.zero_grad(set_to_none=True)
    a_ref.backward()

    for name, param in ref.named_parameters():
        expected = torch.zeros_like(param) if param.grad is None else param.grad
        assert torch.allclose(routed[name], expected, atol=1e-6), name
