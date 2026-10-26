"""Shared settings, boundary policies and CLI parity without model/API calls."""
import csv
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ML1 import matcher as matcher_module, match_cv, match_resume
from ML1.cli import match_processed_resume, matching_config
from ML1.config import EMBEDDING_DIMENSION, MatchingConfig
from ML1.job_assistant import JobAssistant
from ML1.skill_similarity import calculate_skill_similarity
from ML1.vacancy_index import read_vacancies, save_index


class MatchingConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.csv = self.root / 'jobs.csv'
        self.matrix = self.root / 'vectors.npy'
        with self.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['vacancy_id', 'text', 'title', 'skills'])
            writer.writeheader()
            writer.writerows([
                {'vacancy_id': '001', 'text': 'Python job', 'title': 'Backend', 'skills': 'python'},
                {'vacancy_id': '002', 'text': 'Marketing job', 'title': 'Marketing', 'skills': ''},
            ])
        frame, checksum = read_vacancies(self.csv)
        save_index(frame, checksum, np.eye(2, EMBEDDING_DIMENSION), self.matrix)
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(matcher_module, 'VACANCIES_FILE', self.csv))
        stack.enter_context(patch.object(matcher_module, 'EMBEDDINGS_FILE', self.matrix))
        self.encode = stack.enter_context(patch.object(matcher_module, 'encode_text', return_value=np.eye(1, EMBEDDING_DIMENSION)[0]))
        stack.enter_context(redirect_stdout(StringIO()))

    def test_invalid_alpha_is_rejected_before_loading_index(self):
        for value in (-0.1, 1.1, float('nan'), float('inf'), True, '0.4', None):
            with self.subTest(value=value), patch.object(matcher_module, 'read_vacancies') as read:
                with self.assertRaisesRegex(ValueError, 'alpha'):
                    matcher_module.JobMatcher(alpha=value)
                read.assert_not_called()

    def test_invalid_confidence_target_always_rejected_even_without_skills(self):
        for value in (0, -1, float('nan'), float('inf'), True, '5', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'confidence_target'):
                    MatchingConfig(confidence_target=value)
                with self.assertRaisesRegex(ValueError, 'confidence_target'):
                    calculate_skill_similarity([], [], confidence_target=value)

    def test_invalid_top_k_rejected_before_encoding(self):
        matcher = matcher_module.JobMatcher()
        for value in (0, -1, 1.5, True, '10', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'top_k'):
                    matcher.match('Python', top_k=value)
        self.encode.assert_not_called()

    def test_empty_resume_is_rejected_without_encoder_or_index_loading(self):
        matcher = matcher_module.JobMatcher()
        for text in ('', ' \n\t', None, 123):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, 'non-empty string'):
                    matcher.match(text)
                with patch('ML1.cli.JobMatcher') as constructor:
                    with self.assertRaisesRegex(ValueError, 'non-empty string'):
                        match_processed_resume(text)
                    constructor.assert_not_called()
        self.encode.assert_not_called()

    def test_defaults_preserve_baseline_and_emit_actual_metadata(self):
        result = match_processed_resume('Python developer', ['python'])[0]
        self.assertAlmostEqual(result['skill_score'], 0.2)
        self.assertAlmostEqual(result['hybrid_score'], 0.68)
        metadata = result['matching_metadata']
        self.assertEqual(metadata['alpha'], 0.4)
        self.assertEqual(metadata['confidence_target'], 5)
        self.assertEqual(metadata['top_k'], 10)
        self.assertEqual(len(metadata['index_version']), 64)

    def test_custom_confidence_and_alpha_endpoints(self):
        for alpha in (0, 0.65, 1):
            with self.subTest(alpha=alpha):
                result = match_processed_resume('Python', ['python'], MatchingConfig(alpha, 2, 1))[0]
                self.assertAlmostEqual(result['skill_score'], 0.5)
                self.assertAlmostEqual(result['hybrid_score'], alpha * 0.5 + (1 - alpha))
                self.assertEqual(result['matching_metadata']['alpha'], alpha)
                self.assertEqual(result['matching_metadata']['confidence_target'], 2)
                self.assertEqual(result['matching_metadata']['top_k'], 1)

    def test_zero_skills_keep_fixed_weights_and_explicit_empty_list(self):
        with patch.object(matcher_module, 'extract_skills') as extractor:
            results = match_processed_resume('Python', [])
            extractor.assert_not_called()
        self.assertEqual(results[0]['skill_score'], 0)
        self.assertEqual(results[0]['hybrid_score'], 0.6)
        self.assertTrue(all(result['skill_score'] == 0 for result in results))
        self.assertEqual(results[0]['matching_metadata']['zero_skill_policy'], 'zero_skill_score_fixed_weights')

    def test_missing_skills_trigger_extraction_and_exact_text_is_preserved(self):
        text = '  Python\n developer  '
        with patch.object(matcher_module, 'extract_skills', return_value=['python']) as extractor:
            match_processed_resume(text)
        extractor.assert_called_once_with(text)
        self.encode.assert_called_once_with(text)

    def test_top_k_larger_than_corpus_returns_all_rows(self):
        self.assertEqual(len(match_processed_resume('Python', [], MatchingConfig(top_k=100))), 2)

    def test_empty_corpus_is_explicit_error(self):
        self.csv.write_text('vacancy_id,text,title,skills\n')
        with self.assertRaisesRegex(ValueError, 'dataset is empty'):
            matcher_module.JobMatcher()
        self.encode.assert_not_called()

    def test_both_cli_entrypoints_produce_identical_results(self):
        text = '  Python developer  '
        args = ['--alpha', '0.65', '--confidence-target', '3', '--top-k', '1']
        captured = []
        def run(*values, **kwargs):
            result = match_processed_resume(*values, **kwargs)
            captured.append(result)
            return result
        with patch.object(match_resume, 'match_processed_resume', side_effect=run), \
             patch('builtins.input', side_effect=[text, '']):
            match_resume.main(args)
        pdf = self.root / 'fixture.pdf'
        pdf.touch()
        parser_module = types.ModuleType('NLP.cv_parser')
        parser_module.parse_cv = Mock(return_value={'text': text, 'skills': ['python']})
        with patch.dict('sys.modules', {'NLP.cv_parser': parser_module}), \
             patch.object(matcher_module, 'extract_skills', return_value=['python']), \
             patch.object(match_cv, 'match_processed_resume', side_effect=run), \
             patch.object(match_cv, 'select_vacancy', return_value=None):
            # Match text extraction policy for the pasted-text invocation above.
            match_cv.main([str(pdf), *args])
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0], captured[1])
        self.assertEqual(captured[0][0]['matching_metadata']['alpha'], 0.65)

    def test_both_cli_defaults_and_invalid_settings_match(self):
        for module, positional in ((match_resume, []), (match_cv, ['fixture.pdf'])):
            parser = module.make_parser()
            self.assertEqual(matching_config(parser, parser.parse_args(positional)), MatchingConfig())
            for args in (['--alpha', 'nan'], ['--confidence-target', '0'], ['--top-k', '-1']):
                with self.subTest(module=module.__name__, args=args), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        matching_config(parser, parser.parse_args([*positional, *args]))
                    self.assertEqual(error.exception.code, 2)

    def test_assistant_uses_selected_result_weights_without_api(self):
        vacancy = match_processed_resume('Python', ['python'], MatchingConfig(alpha=0.65))[0]
        assistant = JobAssistant.__new__(JobAssistant)
        assistant.vacancy = vacancy
        assistant.resume_text = 'Python'
        assistant.resume_skills = ['python']
        context = assistant._build_context()
        self.assertIn('0.65 * skill_score + 0.35 * semantic_score', context)
        self.assertIn('confidence_target = 5', context)
        self.assertIn(vacancy['matching_metadata']['index_version'], context)
        assistant.vacancy = {k: v for k, v in vacancy.items() if k != 'matching_metadata'}
        self.assertIn('configuration unavailable', assistant._build_context())


if __name__ == '__main__':
    unittest.main()
