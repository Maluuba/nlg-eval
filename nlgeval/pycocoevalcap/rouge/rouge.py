#!/usr/bin/env python
# 
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np
import itertools
import pdb

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

def _get_ngrams(n, text):
    """TAKEN FROM https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
    
    Calcualtes n-grams.
    Args:
        n: which n-grams to calculate
        text: An array of tokens
    Returns:
        A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _split_into_words(sentences):
    """TAKEN FROM https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
    
    Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))

def _get_word_ngrams(n, sentences):
    """TAKEN FROM https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
    
    Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)

def rouge_n(evaluated_sentences, reference_sentences, n=2):
    """ TAKEN FROM https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py

    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    Args:
        evaluated_sentences: The sentences that have been picked by the summarizer
        reference_sentences: The sentences from the referene set
        n: Size of ngram.  Defaults to 2.
    Returns:
        A tuple (f1, precision, recall) for ROUGE-N
    Raises:
        ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    # return overlapping_count / reference_count
    return f1_score, precision, recall

class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self, n=2):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2
        self._n = n

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: list of str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate)==1)	
        assert(len(refs)>0)         
        prec = []
        rec = []

        # Compute ROUGE-n scores
        rouge_n_scores = []
        for n in range(1, self._n + 1):
            f_score, _, _ = rouge_n(candidate, refs, n)
            rouge_n_scores.append(f_score)

        # split into tokens
        token_c = candidate[0].split(" ")
    	
        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            score = 0.0
        return rouge_n_scores + [score]

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py 
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values 
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        score_type = []
        for s_idx, s_type in enumerate(score[0]):
            score_type.append([s[s_idx] for s in score])

        average_score = [np.mean(np.array(s)) for s in score_type]
        return average_score, [np.array(s) for s in score_type]

    def method(self):
        return "Rouge"
