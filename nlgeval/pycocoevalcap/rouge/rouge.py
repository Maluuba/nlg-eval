#!/usr/bin/env python
# 
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

from pyrouge import Rouge155
import shutil

import logging
import os

__cur_path = os.path.dirname(os.path.realpath(__file__))
OFFICIAL_PATH = os.path.join(__cur_path, "official_rouge/")
MODEL_PATH = os.path.join(__cur_path, "tmp/model/")
GOLD_PATH = os.path.join(__cur_path, "tmp/gold/")

class Rouge():
    '''
    Class for computing ROUGE scores for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self, score_type="f_score", rouge_dir=OFFICIAL_PATH, model_tmp=MODEL_PATH, gold_tmp=GOLD_PATH):
        self.rouge155 = Rouge155(rouge_dir, log_level=logging.ERROR)
        self.model_tmp = model_tmp
        self.gold_tmp = gold_tmp

        # Setup rouge155
        self.rouge155.system_dir = self.model_tmp
        self.rouge155.model_dir = self.gold_tmp
        self.rouge155.system_filename_pattern = "rouge.(\d+).txt"
        self.rouge155.model_filename_pattern = "rouge.[A-Z].#ID#.txt"

        # Score type to be returned
        if score_type not in ['f_score', 'recall', 'precision']:
            raise ValueError("Score type must be either 'f_score', 'precision', or 'recall'. Given : {}".format(score_type))
        self.score_type = score_type

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE scores given one candidate and references.
        :param candidate: list of 1 str : candidate sentence to be evaluated
        :param refs: list of str : reference sentences to be evaluated
        :returns recall, precision, f_score: list of float for recall, precision and f score
        """
        assert(len(candidate)==1)	
        assert(len(refs)>0)

        # Write candidate and refs to temp dir for ROUGE155
        with open(os.path.join(self.model_tmp, "rouge.001.txt"), 'w') as f:
            f.write(candidate[0])
        for i, r in enumerate(refs):
            with open(os.path.join(self.gold_tmp, "rouge.{}.001.txt".format(chr(65 + i))), 'w') as f:
                f.write(r)

        # Run the official script
        output = self.rouge155.convert_and_evaluate()
        r = self.rouge155.output_to_dict(output)

        # Order it : Rouge N (1, 2, 3, 4), L, W, S*, SU*
        recall = [r['rouge_1_recall'], r['rouge_2_recall'], r['rouge_3_recall'],
                  r['rouge_4_recall'], r['rouge_l_recall'], r['rouge_w_1.2_recall'],
                  r['rouge_s*_recall'], r['rouge_su*_recall']]
        precision = [r['rouge_1_precision'], r['rouge_2_precision'], r['rouge_3_precision'],
                  r['rouge_4_precision'], r['rouge_l_precision'], r['rouge_w_1.2_precision'],
                  r['rouge_s*_precision'], r['rouge_su*_precision']]
        f_score = [r['rouge_1_f_score'], r['rouge_2_f_score'], r['rouge_3_f_score'],
                  r['rouge_4_f_score'], r['rouge_l_f_score'], r['rouge_w_1.2_f_score'],
                  r['rouge_s*_f_score'], r['rouge_su*_f_score']]

        # Once we got the score, don't forget to remove tmp files
        shutil.rmtree(self.model_tmp) 
        shutil.rmtree(self.gold_tmp) 
        os.makedirs(self.model_tmp)
        os.makedirs(self.gold_tmp)

        return recall, precision, f_score

    def calc_scores(self, gts, res, imgIds):
        """
        Compute ROUGE scores given one candidate and references.
        :param candidate: list of 1 str : candidate sentence to be evaluated
        :param refs: list of str : reference sentences to be evaluated
        :returns recall, precision, f_score: list of float for recall, precision and f score
        """
        for j, idx in enumerate(imgIds):
            candidate = res[idx]
            refs  = gts[idx]

            assert(len(candidate)==1)	
            assert(len(refs)>0)

            # Write candidate and refs to temp dir for ROUGE155
            with open(os.path.join(self.model_tmp, "rouge.{}.txt".format(j + 1)), 'w', encoding='utf-8') as f:
                f.write(candidate[0])
            for i, r in enumerate(refs):
                with open(os.path.join(self.gold_tmp, "rouge.{}.{}.txt".format(chr(65 + i), j + 1)), 'w', encoding='utf-8') as f:
                    f.write(r + "\n")

        # Run the official script
        output = self.rouge155.convert_and_evaluate()
        r = self.rouge155.output_to_dict(output)

        # Order it : Rouge N (1, 2, 3, 4), L, W, S*, SU*
        recall = [r['rouge_1_recall'], r['rouge_2_recall'], r['rouge_3_recall'],
                  r['rouge_4_recall'], r['rouge_l_recall'], r['rouge_w_1.2_recall'],
                  r['rouge_s*_recall'], r['rouge_su*_recall']]
        precision = [r['rouge_1_precision'], r['rouge_2_precision'], r['rouge_3_precision'],
                  r['rouge_4_precision'], r['rouge_l_precision'], r['rouge_w_1.2_precision'],
                  r['rouge_s*_precision'], r['rouge_su*_precision']]
        f_score = [r['rouge_1_f_score'], r['rouge_2_f_score'], r['rouge_3_f_score'],
                  r['rouge_4_f_score'], r['rouge_l_f_score'], r['rouge_w_1.2_f_score'],
                  r['rouge_s*_f_score'], r['rouge_su*_f_score']]

        # Once we got the score, don't forget to remove tmp files
        shutil.rmtree(self.model_tmp) 
        shutil.rmtree(self.gold_tmp) 
        os.makedirs(self.model_tmp)
        os.makedirs(self.gold_tmp)

        return recall, precision, f_score

    def compute_score(self, gts, res):
        """
        Computes Rouge score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py 
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values 
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: list of float (mean ROUGE score computed by averaging scores for each type of ROUGE)
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        recall, precision, f_score = self.calc_scores(gts, res, imgIds)

        if self.score_type == 'recall':
            score = recall
        elif self.score_type == 'precision':
            score = precision
        elif self.score_type == 'f_score':
            score = f_score

        # Return Rouge 1, 2, 3, 4, L, W, S*, SU*
        return score, [None for s in score]

    def method(self):
        return "Rouge"
