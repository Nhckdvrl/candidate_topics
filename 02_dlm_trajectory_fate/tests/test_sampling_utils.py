import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from generate_fates import chosen_token_probability, get_num_transfer_tokens


def test_chosen_probability_matches_softmax_gather():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 17)
    chosen = torch.randint(0, 17, (2, 5))
    got = chosen_token_probability(logits, chosen)
    ref = torch.softmax(logits.float(), dim=-1).gather(
        -1, chosen.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)


def test_transfer_schedule_conserves_masks():
    mask = torch.tensor([[True, True, True, True, True]])
    transfers = get_num_transfer_tokens(mask, 3)
    assert transfers.tolist() == [[2, 2, 1]]
    assert int(transfers.sum()) == 5
