import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


SCRIPT_DIR = Path(__file__).resolve().parent


def is_dataset_root(path):
    return (path / 'train.csv').is_file() and (path / 'test_images').is_dir()


def find_dataset_root(configured_root=None):
    candidates = []
    if configured_root:
        candidates.append(Path(configured_root).expanduser())
    if os.environ.get('EUROSAT_DIR'):
        candidates.append(Path(os.environ['EUROSAT_DIR']).expanduser())

    for base in (SCRIPT_DIR, *SCRIPT_DIR.parents, Path.cwd(), *Path.cwd().parents):
        candidates.extend([base / 'DATASET', base / 'dataset', base])

    visited = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in visited:
            visited.add(resolved)
            if is_dataset_root(resolved):
                return resolved

    raise FileNotFoundError(
        'Dataset not found. Keep DATASET beside this script or pass --dataset-dir '
        'or set EUROSAT_DIR.'
    )


def decode_image(path):
    return tf.io.decode_jpeg(tf.io.read_file(path), channels=3)


def main():
    parser = argparse.ArgumentParser(
        description='Generate EuroSAT competition predictions from a trained model.'
    )
    parser.add_argument('--dataset-dir', type=Path, help='Dataset root containing test_images/')
    parser.add_argument(
        '--model',
        type=Path,
        default=SCRIPT_DIR / 'artifacts' / 'eurosat_cnn.keras',
        help='Path to the trained Keras model.',
    )
    parser.add_argument(
        '--class-names',
        type=Path,
        default=SCRIPT_DIR / 'artifacts' / 'class_names.json',
        help='JSON list mapping model output indices to class labels.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output CSV path. Defaults to submission.csv beside the model.',
    )
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    dataset_root = find_dataset_root(args.dataset_dir)
    test_dir = dataset_root / 'test_images'
    test_items = sorted(
        (path for path in test_dir.iterdir() if path.suffix.lower() == '.jpg'),
        key=lambda path: int(path.stem),
    )
    if not test_items:
        raise FileNotFoundError(f'No .jpg test images found in {test_dir}')
    test_ids = [int(path.stem) for path in test_items]
    if test_ids != list(range(len(test_items))):
        raise ValueError('Expected test image filenames to be numbered consecutively from 00000.')

    if not args.model.is_file():
        raise FileNotFoundError(f'Trained model not found: {args.model}')
    if not args.class_names.is_file():
        raise FileNotFoundError(f'Class mapping not found: {args.class_names}')

    with args.class_names.open(encoding='utf-8') as file:
        class_names = json.load(file)
    if not isinstance(class_names, list) or not all(
        isinstance(class_name, str) for class_name in class_names
    ):
        raise ValueError('Class mapping must be a JSON list of class names.')

    test_dataset = tf.data.Dataset.from_tensor_slices(
        np.asarray([str(path) for path in test_items], dtype=str)
    )
    test_dataset = test_dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(args.model)
    if model.output_shape[-1] != len(class_names):
        raise ValueError('Class mapping length does not match the model output dimension.')
    probabilities = model.predict(test_dataset, verbose=1)
    predicted_class_indices = probabilities.argmax(axis=1).astype(int)
    submission = pd.DataFrame(
        {
            'id': [f'test_{image_id + 1:05d}' for image_id in test_ids],
            'label': predicted_class_indices,
        }
    )

    output_path = args.output or args.model.parent / 'submission.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f'Wrote {len(submission)} predictions to {output_path}')


if __name__ == '__main__':
    main()