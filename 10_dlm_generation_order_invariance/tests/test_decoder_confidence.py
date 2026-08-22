from types import SimpleNamespace

import torch

from instrumented_llada import decode_fixed_slots


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and text in "123456789":
            return [100 + int(text)]
        return [300 + (ord(ch) % 500) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        if len(ids) == 1 and 101 <= ids[0] <= 109:
            return str(ids[0] - 100)
        return "x"


class FakeModel:
    device = torch.device("cpu")

    def __call__(self, x):
        vocab = 1200
        logits = torch.full((1, x.shape[1], vocab), -5.0)
        masks = torch.nonzero(x[0].eq(999), as_tuple=False).flatten().tolist()
        for rank, pos in enumerate(masks):
            if rank == 0:
                # Among digits, token 1 looks overwhelmingly best. But a native
                # non-digit token has much larger logit, so the digit's absolute
                # full-vocabulary probability is low.
                logits[0, pos, 101] = 5.0
                logits[0, pos, 150] = 10.0
            else:
                # Digit 2 is less dominant within the digit subset, but has much
                # higher full-vocabulary probability than cell A's best digit.
                logits[0, pos, 101:110] = 3.9
                logits[0, pos, 102] = 4.0
                logits[0, pos, 150] = 4.1
        return SimpleNamespace(logits=logits)


def test_scheduler_uses_full_vocab_probability_not_digit_renormalization():
    puzzle = [1] * 81
    puzzle[0] = 0
    puzzle[1] = 0
    result = decode_fixed_slots(
        FakeModel(),
        FakeTokenizer(),
        puzzle,
        "dummy",
        mask_id=999,
        remasking="low_confidence",
        temperature=0.0,
        seed=0,
    )
    assert result.finalization_step[1] == 1
    assert result.finalization_step[0] == 2
    # Native scheduler would initially prefer the strong non-digit at cell 0,
    # demonstrating that the grammar projection is explicitly measurable.
    assert result.native_scheduler_pick_same[1] is False
