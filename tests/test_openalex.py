"""Tests for quarry.etl.openalex utilities."""

from quarry.etl.openalex import reconstruct_abstract, work_id_to_int


class TestReconstructAbstract:
    def test_basic(self):
        inv_idx = {"This": [0], "is": [1], "a": [2], "test": [3]}
        assert reconstruct_abstract(inv_idx) == "This is a test"

    def test_repeated_word(self):
        inv_idx = {"the": [0, 2], "cat": [1], "sat": [3]}
        assert reconstruct_abstract(inv_idx) == "the cat the sat"

    def test_real_oa_sample(self):
        # Simplified real OA abstract_inverted_index structure
        inv_idx = {
            "Abstract": [0],
            "We": [1],
            "present": [2],
            "a": [3, 8],
            "method": [4],
            "for": [5],
            "analyzing": [6],
            "citation": [7],
            "network.": [9],
        }
        result = reconstruct_abstract(inv_idx)
        assert (
            result == "Abstract We present a method for analyzing citation a network."
        )

    def test_empty_input(self):
        assert reconstruct_abstract({}) == ""

    def test_none_input(self):
        assert reconstruct_abstract(None) == ""

    def test_single_word(self):
        assert reconstruct_abstract({"Hello": [0]}) == "Hello"


class TestWorkIdToInt:
    def test_short_id(self):
        assert work_id_to_int("W2741809807") == 2741809807

    def test_full_url(self):
        assert work_id_to_int("https://openalex.org/W2741809807") == 2741809807

    def test_numeric_only(self):
        assert work_id_to_int("W123") == 123

    def test_large_id(self):
        # OA IDs can be ~10^10
        assert work_id_to_int("W10000000000") == 10000000000
