"""Index provenance regressions; no model, API or private data required."""
import csv
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ML1 import build_vacancy_embeddings as builder
from ML1.config import EMBEDDING_DIMENSION
from ML1.vacancy_index import (
    load_index, manifest_path, read_vacancies, save_index, sha256,
)


class IndexTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.csv = self.root / "vacancies.csv"
        self.matrix = self.root / "vectors.npy"
        self.rows = [
            {"vacancy_id": "001", "text": "Python backend", "title": "Backend", "skills": "python"},
            {"vacancy_id": "NA", "text": "Frontend JS", "title": "Frontend", "skills": "javascript"},
        ]
        self.write_csv(self.rows)
        self.vectors = np.eye(2, EMBEDDING_DIMENSION, dtype=np.float32)
        self.frame, self.csv_hash = read_vacancies(self.csv)
        self.metadata = save_index(self.frame, self.csv_hash, self.vectors, self.matrix)

    def write_csv(self, rows):
        with self.csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def load(self):
        frame, checksum = read_vacancies(self.csv)
        return load_index(frame, checksum, self.matrix)

    def change_metadata(self, **changes):
        value = json.loads(manifest_path(self.matrix).read_text())
        value.update(changes)
        manifest_path(self.matrix).write_text(json.dumps(value))

    def test_round_trip_and_manifest(self):
        matrix, metadata = self.load()
        np.testing.assert_array_equal(matrix, self.vectors)
        self.assertEqual([r['vacancy_id'] for r in metadata['rows']], ['001', 'NA'])
        self.assertEqual(metadata['csv_sha256'], sha256(self.csv.read_bytes()))
        self.assertEqual(metadata['matrix_sha256'], sha256(self.matrix.read_bytes()))
        self.assertEqual(metadata['rows'][0]['text_sha256'], sha256(b'Python backend'))
        self.assertEqual(metadata['shape'], [2, 384])
        self.assertEqual(len(metadata['index_version']), 64)

    def test_csv_reordering_with_unchanged_row_count_is_rejected(self):
        self.write_csv(self.rows[::-1])
        with self.assertRaisesRegex(ValueError, "row order.*Rebuild"):
            self.load()

    def test_changed_source_fields_are_rejected(self):
        for field in ('vacancy_id', 'text', 'title', 'skills'):
            with self.subTest(field=field):
                self.write_csv([{**self.rows[0], field: 'changed'}, self.rows[1]])
                with self.assertRaisesRegex(ValueError, "CSV content"):
                    self.load()

    def test_changed_vectors_with_same_shape_are_rejected(self):
        np.save(self.matrix, self.vectors[::-1])
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.load()

    def test_interrupted_manifest_publish_is_detected(self):
        real_replace = os.replace
        def replace(source, target):
            if Path(target) == manifest_path(self.matrix):
                raise OSError('simulated interrupted publication')
            return real_replace(source, target)
        with patch('ML1.vacancy_index.os.replace', side_effect=replace):
            with self.assertRaises(OSError):
                save_index(self.frame, self.csv_hash, self.vectors[::-1], self.matrix)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.load()
        self.assertEqual(set(self.root.iterdir()), {self.csv, self.matrix, manifest_path(self.matrix)})

    def test_missing_matrix_is_rejected(self):
        self.matrix.unlink()
        with self.assertRaisesRegex(ValueError, 'matrix is missing'):
            self.load()

    def test_missing_manifest_is_rejected(self):
        manifest_path(self.matrix).unlink()
        with self.assertRaisesRegex(ValueError, "manifest is missing.*Rebuild"):
            self.load()

    def test_corrupt_or_unsupported_manifest_is_rejected(self):
        for value in ('broken json', '[]', '{"schema_version": 2}'):
            with self.subTest(value=value):
                manifest_path(self.matrix).write_text(value)
                with self.assertRaisesRegex(ValueError, 'manifest'):
                    self.load()

    def test_model_and_normalization_settings_must_match(self):
        for changes in ({'model_name': 'other-model'}, {'normalize_embeddings': False}):
            with self.subTest(changes=changes):
                manifest_path(self.matrix).write_text(json.dumps(self.metadata))
                self.change_metadata(**changes)
                with self.assertRaisesRegex(ValueError, 'model/settings'):
                    self.load()

    def test_manifest_row_mapping_must_match(self):
        self.change_metadata(rows=self.metadata['rows'][::-1])
        with self.assertRaisesRegex(ValueError, 'row order'):
            self.load()

    def test_manifest_shape_and_dtype_must_match(self):
        for changes in ({'shape': [2, 2]}, {'dtype': 'float64'}):
            with self.subTest(changes=changes):
                manifest_path(self.matrix).write_text(json.dumps(self.metadata))
                self.change_metadata(**changes)
                with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                    self.load()

    def test_version_fingerprint_must_match(self):
        self.change_metadata(index_version='wrong')
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            self.load()

    def test_invalid_matrices_cannot_replace_valid_index(self):
        bad = [self.vectors[0], self.vectors[:1], self.vectors[:, :2],
               self.vectors.astype(int), self.vectors * 2, self.vectors * np.nan]
        original = self.matrix.read_bytes()
        for vectors in bad:
            with self.subTest(shape=vectors.shape):
                with self.assertRaises(ValueError):
                    save_index(self.frame, self.csv_hash, vectors, self.matrix)
                self.assertEqual(self.matrix.read_bytes(), original)

    def test_numeric_looking_text_is_preserved(self):
        self.write_csv([{**self.rows[0], 'text': '00123'}, {**self.rows[1], 'text': 'null'}])
        frame, _ = read_vacancies(self.csv)
        self.assertEqual(frame['text'].tolist(), ['00123', 'null'])

    def test_build_produces_loadable_pair_with_fake_encoder(self):
        with patch.object(builder, 'VACANCIES_FILE', self.csv), \
             patch.object(builder, 'OUTPUT_FILE', self.matrix), \
             patch.object(builder, 'encode_texts', return_value=self.vectors) as encoder, \
             redirect_stdout(StringIO()):
            builder.main()
        encoder.assert_called_once_with(['Python backend', 'Frontend JS'], batch_size=32)
        np.testing.assert_array_equal(self.load()[0], self.vectors)

    def test_csv_edit_during_encoding_does_not_publish(self):
        original = self.matrix.read_bytes()
        def encode(*args, **kwargs):
            self.write_csv(self.rows[::-1])
            return self.vectors
        with patch.object(builder, 'VACANCIES_FILE', self.csv), \
             patch.object(builder, 'OUTPUT_FILE', self.matrix), \
             patch.object(builder, 'encode_texts', side_effect=encode), \
             redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError, 'changed while building'):
                builder.main()
        self.assertEqual(self.matrix.read_bytes(), original)

    def test_rebuild_after_reordering_is_valid(self):
        self.write_csv(self.rows[::-1])
        frame, checksum = read_vacancies(self.csv)
        rebuilt = save_index(frame, checksum, self.vectors[::-1], self.matrix)
        self.assertNotEqual(rebuilt['index_version'], self.metadata['index_version'])
        matrix, _ = self.load()
        np.testing.assert_array_equal(matrix, self.vectors[::-1])


if __name__ == '__main__':
    unittest.main()
