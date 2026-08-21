from memory_interference.scoring import longest_common_prefix


def test_longest_common_prefix():
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([], [1]) == 0
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2


def test_git_blob_checksum_formula():
    from scripts.download_data import git_blob_sha1
    assert git_blob_sha1(b"test content\n") == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"
