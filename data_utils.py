"""Preprocessing shared by training and prediction."""
import numpy as np


def vectorize_sequences(sequences, dimension=2000):
    """R-compatible multi-hot encoding: word ID i maps to column i-1.

    Ignore padding ID 0; repeated words remain binary. Word order is discarded.
    """
    if dimension < 1:
        raise ValueError('dimension must be positive')
    result = np.zeros((len(sequences), dimension), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        indices = np.asarray(sequence, dtype=np.int64)
        indices = indices[(indices > 0) & (indices <= dimension)]
        result[row, indices - 1] = 1
    return result


def split_indices(size, validation_size=5000, seed=94610):
    """Split only the original training set; never sample from test examples."""
    if not 0 < validation_size < size:
        raise ValueError('validation_size must be between 1 and size-1')
    validation = np.random.default_rng(seed).choice(size, validation_size, replace=False)
    training = np.setdiff1d(np.arange(size), validation)
    return training, validation


def decode_review(sequence, word_index):
    reverse = {index + 3: word for word, index in word_index.items()}
    reverse.update({0: '<PAD>', 1: '<START>', 2: '<OOV>', 3: '<UNUSED>'})
    return ' '.join(reverse.get(int(index), '<?>') for index in sequence)
