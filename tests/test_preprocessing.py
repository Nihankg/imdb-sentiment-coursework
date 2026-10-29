import unittest
import numpy as np
from data_utils import vectorize_sequences, split_indices, decode_review


class PreprocessingTests(unittest.TestCase):
    def test_padding_duplicates_and_r_indexing(self):
        encoded = vectorize_sequences([[0, 1, 2, 2, 5, 6], []], 5)
        np.testing.assert_array_equal(encoded, [[1, 1, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.assertEqual(encoded.dtype, np.float32)

    def test_split_is_disjoint_and_complete(self):
        train, val = split_indices(25000)
        self.assertEqual((len(train), len(val)), (20000, 5000))
        self.assertEqual(len(np.intersect1d(train, val)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(25000))
        np.testing.assert_array_equal(val, split_indices(25000)[1])

    def test_decoding_uses_keras_offset(self):
        self.assertEqual(decode_review([1, 4, 2, 0], {'good': 1}),
                         '<START> good <OOV> <PAD>')


if __name__ == '__main__':
    unittest.main()
