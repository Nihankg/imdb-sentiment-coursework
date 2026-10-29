"""Train a feed-forward sentiment classifier adapted from R/Keras coursework."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
from data_utils import vectorize_sequences, split_indices, decode_review


@dataclass
class Config:
    num_words: int = 2000
    hidden_units: tuple = (512, 16)
    batch_size: int = 128
    epochs: int = 7
    learning_rate: float = 0.0001
    patience: int = 10
    validation_size: int = 5000
    seed: int = 94610


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/baseline'))
    parser.add_argument('--hidden-units', type=int, nargs='+', default=[512, 16])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num-words', type=int, default=2000)
    parser.add_argument('--validation-size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=94610)
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Evaluate held-out test data after finalizing hyperparameters')
    args = parser.parse_args()
    c = Config(args.num_words, tuple(args.hidden_units), args.batch_size, args.epochs,
               args.learning_rate, args.patience, args.validation_size, args.seed)
    if min(c.num_words, c.batch_size, c.epochs, *c.hidden_units) < 1:
        parser.error('Vocabulary, batch size, epochs and hidden units must be positive')
    if c.learning_rate <= 0 or c.patience < 0:
        parser.error('Learning rate must be positive; patience must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output for each run')

    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(c.seed)
    tf.config.experimental.enable_op_determinism()
    started = time.monotonic()
    (train_sequences, labels), (test_sequences, test_labels) = (
        tf.keras.datasets.imdb.load_data(num_words=c.num_words))
    train_ids, val_ids = split_indices(len(labels), c.validation_size, c.seed)
    train_x = vectorize_sequences(train_sequences[train_ids], c.num_words)
    val_x = vectorize_sequences(train_sequences[val_ids], c.num_words)
    train_y = np.asarray(labels[train_ids], dtype=np.float32)
    val_y = np.asarray(labels[val_ids], dtype=np.float32)

    model = tf.keras.Sequential(
        [tf.keras.Input(shape=(c.num_words,))]
        + [tf.keras.layers.Dense(units, activation='relu') for units in c.hidden_units]
        + [tf.keras.layers.Dense(1, activation='sigmoid')], name='imdb_ffnn')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=c.learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    args.output.mkdir(parents=True, exist_ok=True)
    model.summary(print_fn=lambda text: print(text, flush=True))
    summary = []
    model.summary(print_fn=summary.append)
    (args.output / 'model_summary.txt').write_text('\n'.join(summary))
    (args.output / 'config.json').write_text(json.dumps(asdict(c), indent=2))
    np.savez(args.output / 'split_indices.npz', training=train_ids, validation=val_ids)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=c.patience, restore_best_weights=True)
    history = model.fit(
        train_x, train_y, validation_data=(val_x, val_y),
        epochs=c.epochs, batch_size=c.batch_size,
        callbacks=[early_stop, tf.keras.callbacks.CSVLogger(str(args.output / 'history.csv'))],
        verbose=2)
    # Metrics are for the weights restored from the best validation-loss epoch.
    validation_metrics = model.evaluate(val_x, val_y, verbose=0, return_dict=True)
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'Keras IMDB, default dataset shuffle seed 113',
        'training_examples': len(train_ids), 'validation_examples': len(val_ids),
        'epochs_completed': len(history.history['loss']), 'best_epoch': best_epoch,
        'validation': validation_metrics, 'test': None,
        'config': asdict(c), 'runtime': {'python': platform.python_version(),
                                      'tensorflow': tf.__version__, 'numpy': np.__version__},
        'validation_indices_sha256': hashlib.sha256(val_ids.tobytes()).hexdigest(),
        'training_type': 'random initialization; no pretrained model or fine-tuning'}
    if args.evaluate_test:
        test_x = vectorize_sequences(test_sequences, c.num_words)
        test_y = np.asarray(test_labels, dtype=np.float32)
        report['test'] = model.evaluate(test_x, test_y, verbose=0, return_dict=True)
        report['test_examples'] = len(test_labels)
        probabilities = model.predict(test_x[:10], verbose=0).reshape(-1)
        words = tf.keras.datasets.imdb.get_word_index()
        examples = [{'test_index': i, 'true_label': int(test_labels[i]),
                     'predicted_probability': float(p), 'predicted_label': int(p > 0.5),
                     'decoded_excerpt': decode_review(test_sequences[i][:60], words)}
                    for i, p in enumerate(probabilities)]
        (args.output / 'predictions.json').write_text(json.dumps(examples, indent=2))
    model.save(args.output / 'model.keras')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    epochs = np.arange(1, len(history.history['loss']) + 1)
    for ax, key, title in zip(axes, ['loss', 'accuracy'], ['Binary cross-entropy', 'Accuracy']):
        ax.plot(epochs, history.history[key], label='Training')
        ax.plot(epochs, history.history['val_' + key], label='Validation')
        ax.axvline(best_epoch, color='gray', linestyle=':', label='Selected epoch')
        ax.set(xlabel='Epoch', ylabel=title)
        ax.legend()
    fig.suptitle('IMDB sentiment classification | coursework baseline')
    fig.savefig(args.output / 'training_history.png', dpi=180)
    plt.close(fig)
    report['elapsed_seconds'] = time.monotonic() - started
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
