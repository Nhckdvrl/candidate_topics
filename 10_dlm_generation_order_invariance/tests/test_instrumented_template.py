from instrumented_llada import _exact_digit_token_ids, build_sequence


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Digits are single, unique exact tokens; all other characters are fixed
        # non-mask tokens. This is enough to test cell-slot accounting.
        if len(text) == 1 and text in "123456789":
            return [100 + int(text)]
        return [1000 + ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        if len(ids) == 1 and 101 <= ids[0] <= 109:
            return str(ids[0] - 100)
        return "".join(chr(i - 1000) for i in ids if i >= 1000)


def test_template_has_exactly_one_cell_position_per_cell():
    tok = FakeTokenizer()
    mask_id = 9999
    puzzle = tuple(0 if i in {0, 10, 80} else (i % 9) + 1 for i in range(81))
    seq, digit_ids, cell_positions = build_sequence(tok, [1, 2, 3], puzzle, mask_id)
    assert _exact_digit_token_ids(tok) == digit_ids
    assert len(cell_positions) == 81
    assert len(set(cell_positions)) == 81
    assert {cell_positions[i] for i in (0, 10, 80)} == {j for j, token in enumerate(seq) if token == mask_id}
    for i, value in enumerate(puzzle):
        if value:
            assert seq[cell_positions[i]] == digit_ids[value]
