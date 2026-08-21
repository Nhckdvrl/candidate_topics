from memory_interference.scoring import longest_common_prefix


def test_longest_common_prefix():
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([], [1]) == 0
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2


def test_git_blob_checksum_formula():
    from scripts.download_data import git_blob_sha1
    assert git_blob_sha1(b"test content\n") == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"


def test_score_candidates_uses_causal_next_token_positions():
    import torch
    from types import SimpleNamespace
    from memory_interference.scoring import score_candidates

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def __call__(self, text, add_special_tokens=True):
            table = {"P\n": [1, 2], "P\nA": [1, 2, 3], "P\nB": [1, 2, 4]}
            return {"input_ids": table[text]}

    class TinyModel:
        def __call__(self, input_ids=None, attention_mask=None):
            b, n = input_ids.shape
            logits = torch.zeros((b, n, 8), dtype=torch.float32)
            logits[:, 1, 3] = 5.0
            logits[:, 1, 4] = 1.0
            return SimpleNamespace(logits=logits)

    scores = score_candidates(
        TinyModel(), TinyTokenizer(), "P\n", ["A", "B"], device=torch.device("cpu")
    )
    by_name = {s.candidate: s for s in scores}
    assert by_name["A"].mean_logprob > by_name["B"].mean_logprob
    assert by_name["A"].token_count == 1
    assert by_name["A"].boundary_shift == 0
