import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic12.ablation import _scaled_output, parse_layer_spec, residual_scale_layer


class ToyBlock(torch.nn.Module):
    def forward(self, hidden_states, use_cache=False):
        y = hidden_states + 2.0
        if use_cache:
            return y, "cache"
        return y


class ToyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyBlock()])


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyInner()


def test_residual_scale_endpoints_and_half():
    m = ToyModel()
    x = torch.tensor([[1.0]])
    original = m.model.layers[0](x)
    with residual_scale_layer(m, 0, scale=0.0):
        assert torch.equal(m.model.layers[0](x), x)
    with residual_scale_layer(m, 0, scale=0.5):
        assert torch.equal(m.model.layers[0](x), torch.tensor([[2.0]]))
    with residual_scale_layer(m, 0, scale=1.0):
        assert torch.equal(m.model.layers[0](x), original)


def test_scale_zero_does_not_propagate_discarded_nan_update():
    x = torch.tensor([[1.0]])
    out = torch.tensor([[float("nan")]])
    y = _scaled_output(x, out, 0.0)
    assert torch.equal(y, x)


def test_residual_scale_preserves_aux_output():
    m = ToyModel()
    x = torch.tensor([[1.0]])
    with residual_scale_layer(m, 0, scale=0.0):
        y, cache = m.model.layers[0](x, use_cache=True)
    assert torch.equal(y, x)
    assert cache == "cache"


def test_layer_parser():
    assert parse_layer_spec("all", 4) == [0, 1, 2, 3]
    assert parse_layer_spec("none", 4) == []
    assert parse_layer_spec("0,2-3", 4) == [0, 2, 3]
