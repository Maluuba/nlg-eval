#!/usr/bin/env python
# 
# File Name : bert_scorer.py
#
# Description : Computes BERT score as described by Tianyi Zhang et all (2019)
#
# Creation Date : 2019-07-06
# Author : REMOND Nicolas

from bert_score import score

class BertScore():
    '''
    Class for computing BERT score for a set of candidate sentences
    '''

    def __init__(self, score_type='f_score'):
        # Score type to be returned
        if score_type not in ['f_score', 'recall', 'precision']:
            raise ValueError("Score type must be either 'f_score', 'precision', or 'recall'. Given : {}".format(score_type))
        self.score_type = score_type

    def compute_score(self, gts, res):
        """
        Computes BERT score given a set of reference and candidate sentences for the dataset
        :param res: dict : candidate / test sentences. 
        :param gts: dict : references.
        :returns: average_score: float (mean BERT score computed by averaging scores for all the images), individual scores
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        hyp = [res[id][0] for id in imgIds]
        ref = [gts[id][0] for id in imgIds]    # Take only the first reference
                                               # Because Bert Score support only 1
        assert len(hyp) == len(ref)

        P, R, F1 = score(hyp, ref, bert="bert-base-uncased", no_idf=(len(ref) == 1))

        if self.score_type == 'recall':
            s = R
        elif self.score_type == 'precision':
            s = P
        elif self.score_type == 'f_score':
            s = F1

        return s.mean().item(), s.tolist()