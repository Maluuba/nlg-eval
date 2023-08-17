# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import unittest

import nlgeval
from nlgeval import NLGEval


class TestNlgEval(unittest.TestCase):
    def test_compute_metrics_oo(self):
        # Create the object in the test so that it can be garbage collected once the test is done.
        n = NLGEval()

        # Individual Metrics
        scores = n.compute_individual_metrics(ref=["this is a test",
                                                   "this is also a test"],
                                              hyp="this is a good test")
        self.assertAlmostEqual(0.799999, scores['Bleu_1'], places=5)
        self.assertAlmostEqual(0.632455, scores['Bleu_2'], places=5)
        self.assertAlmostEqual(0.5108729, scores['Bleu_3'], places=5)
        self.assertAlmostEqual(0.0000903602, scores['Bleu_4'], places=5)
        self.assertAlmostEqual(0.44434387, scores['METEOR'], places=5)
        self.assertAlmostEqual(0.9070631, scores['ROUGE_L'], places=5)
        self.assertAlmostEqual(0.0, scores['CIDEr'], places=5)
        self.assertAlmostEqual(0.8375251, scores['SkipThoughtCS'], places=5)
        self.assertAlmostEqual(0.980075, scores['EmbeddingAverageCosineSimilarity'], places=5)
        self.assertEqual(scores['EmbeddingAverageCosineSimilarity'], scores['EmbeddingAverageCosineSimilairty'])
        self.assertAlmostEqual(0.94509, scores['VectorExtremaCosineSimilarity'], places=5)
        self.assertAlmostEqual(0.960771, scores['GreedyMatchingScore'], places=5)
        self.assertEqual(12, len(scores))

        scores = n.compute_metrics(ref_list=[
            [
                "this is one reference sentence for sentence1",
                "this is a reference sentence for sentence2 which was generated by your model"
            ],
            [
                "this is one more reference sentence for sentence1",
                "this is the second reference sentence for sentence2"
            ],
        ],
            hyp_list=[
                "this is the model generated sentence1 which seems good enough",
                "this is sentence2 which has been generated by your model"
            ]
        )
        self.assertAlmostEqual(0.55, scores['Bleu_1'], places=5)
        self.assertAlmostEqual(0.428174, scores['Bleu_2'], places=5)
        self.assertAlmostEqual(0.284043, scores['Bleu_3'], places=5)
        self.assertAlmostEqual(0.201143, scores['Bleu_4'], places=5)
        self.assertAlmostEqual(0.295797, scores['METEOR'], places=5)
        self.assertAlmostEqual(0.522104, scores['ROUGE_L'], places=5)
        self.assertAlmostEqual(1.242192, scores['CIDEr'], places=5)
        self.assertAlmostEqual(0.626149, scores['SkipThoughtCS'], places=5)
        self.assertAlmostEqual(0.88469, scores['EmbeddingAverageCosineSimilarity'], places=5)
        self.assertAlmostEqual(0.568696, scores['VectorExtremaCosineSimilarity'], places=5)
        self.assertAlmostEqual(0.784205, scores['GreedyMatchingScore'], places=5)
        self.assertEqual(12, len(scores))

        # Non-ASCII tests.
        scores = n.compute_individual_metrics(ref=["Test en français.",
                                                   "Le test en français."],
                                              hyp="Le test est en français.")
        self.assertAlmostEqual(0.799999, scores['Bleu_1'], places=5)
        self.assertAlmostEqual(0.632455, scores['Bleu_2'], places=5)
        self.assertAlmostEqual(0.0000051, scores['Bleu_3'], places=5)
        self.assertAlmostEqual(0, scores['Bleu_4'], places=5)
        self.assertAlmostEqual(0.48372379050300296, scores['METEOR'], places=5)
        self.assertAlmostEqual(0.9070631, scores['ROUGE_L'], places=5)
        self.assertAlmostEqual(0.0, scores['CIDEr'], places=5)
        self.assertAlmostEqual(0.9192341566085815, scores['SkipThoughtCS'], places=5)
        self.assertAlmostEqual(0.906562, scores['EmbeddingAverageCosineSimilarity'], places=5)
        self.assertEqual(scores['EmbeddingAverageCosineSimilarity'], scores['EmbeddingAverageCosineSimilairty'])
        self.assertAlmostEqual(0.815158, scores['VectorExtremaCosineSimilarity'], places=5)
        self.assertAlmostEqual(0.940959, scores['GreedyMatchingScore'], places=5)
        self.assertEqual(12, len(scores))

        scores = n.compute_individual_metrics(ref=["テスト"],
                                              hyp="テスト")
        self.assertAlmostEqual(0.99999999, scores['Bleu_1'], places=5)
        self.assertAlmostEqual(1.0, scores['METEOR'], places=3)
        self.assertAlmostEqual(1.0, scores['ROUGE_L'], places=3)
        self.assertAlmostEqual(0.0, scores['CIDEr'], places=3)
        self.assertAlmostEqual(1.0, scores['SkipThoughtCS'], places=3)
        self.assertAlmostEqual(1.0, scores['GreedyMatchingScore'], places=3)
        self.assertEqual(12, len(scores))

    def test_compute_metrics_omit(self):
        n = NLGEval(metrics_to_omit=['Bleu_3', 'METEOR', 'EmbeddingAverageCosineSimilarity'])

        # Individual Metrics
        scores = n.compute_individual_metrics(ref=["this is a test",
                                                   "this is also a test"],
                                              hyp="this is a good test")
        self.assertAlmostEqual(0.799999, scores['Bleu_1'], places=5)
        self.assertAlmostEqual(0.632455, scores['Bleu_2'], places=5)
        self.assertAlmostEqual(0.9070631, scores['ROUGE_L'], places=5)
        self.assertAlmostEqual(0.0, scores['CIDEr'], places=5)
        self.assertAlmostEqual(0.8375251, scores['SkipThoughtCS'], places=5)
        self.assertAlmostEqual(0.94509, scores['VectorExtremaCosineSimilarity'], places=5)
        self.assertAlmostEqual(0.960771, scores['GreedyMatchingScore'], places=5)
        self.assertEqual(7, len(scores))

    def test_compute_metrics_empty(self):
        n = NLGEval()

        # One of the ref is empty
        scores = n.compute_individual_metrics(ref=["this is a test",
                                                   ""],
                                              hyp="this is a good test")
        self.assertAlmostEqual(0.799999, scores['Bleu_1'], places=5)
        self.assertAlmostEqual(0.632455, scores['Bleu_2'], places=5)
        self.assertAlmostEqual(0.5108729, scores['Bleu_3'], places=5)
        self.assertAlmostEqual(0.0000903602, scores['Bleu_4'], places=5)
        self.assertAlmostEqual(0.44434387, scores['METEOR'], places=5)
        self.assertAlmostEqual(0.9070631, scores['ROUGE_L'], places=5)
        self.assertAlmostEqual(0.0, scores['CIDEr'], places=5)
        self.assertAlmostEqual(0.8375251, scores['SkipThoughtCS'], places=5)
        self.assertAlmostEqual(0.980075, scores['EmbeddingAverageCosineSimilarity'], places=5)
        self.assertEqual(scores['EmbeddingAverageCosineSimilarity'], scores['EmbeddingAverageCosineSimilairty'])
        self.assertAlmostEqual(0.94509, scores['VectorExtremaCosineSimilarity'], places=5)
        self.assertAlmostEqual(0.960771, scores['GreedyMatchingScore'], places=5)
        self.assertEqual(12, len(scores))

        # Empty hyp
        scores = n.compute_individual_metrics(ref=["this is a good test"],
                                              hyp="")
        self.assertAlmostEqual(0, scores['Bleu_1'], places=5)
        self.assertAlmostEqual(0, scores['Bleu_2'], places=5)
        self.assertAlmostEqual(0, scores['Bleu_3'], places=5)
        self.assertAlmostEqual(0, scores['Bleu_4'], places=5)
        self.assertAlmostEqual(0, scores['METEOR'], places=5)
        self.assertAlmostEqual(0, scores['ROUGE_L'], places=5)
        self.assertAlmostEqual(0, scores['CIDEr'], places=5)
        self.assertAlmostEqual(0, scores['SkipThoughtCS'], places=5)
        self.assertAlmostEqual(0, scores['EmbeddingAverageCosineSimilarity'], places=5)
        self.assertEqual(scores['EmbeddingAverageCosineSimilarity'], scores['EmbeddingAverageCosineSimilairty'])
        self.assertAlmostEqual(0, scores['VectorExtremaCosineSimilarity'], places=5)
        self.assertAlmostEqual(0, scores['GreedyMatchingScore'], places=5)
        self.assertEqual(12, len(scores))

    def test_compute_metrics(self):
        # The example from the README.
        root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        hypothesis = os.path.join(root_dir, 'examples/hyp.txt')
        references = os.path.join(root_dir, 'examples/ref1.txt'), os.path.join(root_dir, 'examples/ref2.txt')
        scores = nlgeval.compute_metrics(hypothesis, references)
        self.assertAlmostEqual(0.55, scores['Bleu_1'], places=5)
        self.assertAlmostEqual(0.428174, scores['Bleu_2'], places=5)
        self.assertAlmostEqual(0.284043, scores['Bleu_3'], places=5)
        self.assertAlmostEqual(0.201143, scores['Bleu_4'], places=5)
        self.assertAlmostEqual(0.295797, scores['METEOR'], places=5)
        self.assertAlmostEqual(0.522104, scores['ROUGE_L'], places=5)
        self.assertAlmostEqual(1.242192, scores['CIDEr'], places=5)
        self.assertAlmostEqual(0.626149, scores['SkipThoughtCS'], places=5)
        self.assertAlmostEqual(0.88469, scores['EmbeddingAverageCosineSimilarity'], places=5)
        self.assertEqual(scores['EmbeddingAverageCosineSimilarity'], scores['EmbeddingAverageCosineSimilairty'])
        self.assertAlmostEqual(0.568696, scores['VectorExtremaCosineSimilarity'], places=5)
        self.assertAlmostEqual(0.784205, scores['GreedyMatchingScore'], places=5)
        self.assertEqual(12, len(scores))
