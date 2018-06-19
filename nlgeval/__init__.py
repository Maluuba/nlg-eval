# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
from __future__ import print_function

from six.moves import map

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge


def compute_metrics(hypothesis, references, no_overlap=False, no_skipthoughts=False, no_glove=False):
    with open(hypothesis, 'r') as f:
        hyp_list = f.readlines()
    ref_list = []
    for iidx, reference in enumerate(references):
        with open(reference, 'r') as f:
            ref_list.append(f.readlines())
    ref_list = [list(map(str.strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.6f" % (m, sc))
                    ret_scores[m] = sc
            else:
                print("%s: %0.6f" % (method, score))
                ret_scores[method] = score

    if not no_skipthoughts:
        from nlgeval.skipthoughts import skipthoughts
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        model = skipthoughts.load_model()
        encoder = skipthoughts.Encoder(model)
        vector_hyps = encoder.encode([h.strip() for h in hyp_list], verbose=False)
        ref_list_T = np.array(ref_list).T.tolist()
        vector_refs = map(lambda refl: encoder.encode([r.strip() for r in refl], verbose=False), ref_list_T)
        cosine_similarity = list(map(lambda refv: cosine_similarity(refv, vector_hyps).diagonal(), vector_refs))
        cosine_similarity = np.max(cosine_similarity, axis=0).mean()
        print("SkipThoughtsCosineSimilairty: %0.6f" % (cosine_similarity))
        ret_scores['SkipThoughtCS'] = cosine_similarity

    if not no_glove:
        from nlgeval.word2vec.evaluate import eval_emb_metrics
        import numpy as np

        glove_hyps = [h.strip() for h in hyp_list]
        ref_list_T = np.array(ref_list).T.tolist()
        glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
        scores = eval_emb_metrics(glove_hyps, glove_refs)
        print(scores)
        scores = scores.split('\n')
        for score in scores:
            name, value = score.split(':')
            value = float(value.strip())
            ret_scores[name] = value

    return ret_scores


def compute_individual_metrics(ref, hyp, no_overlap=False, no_skipthoughts=False, no_glove=False):
    assert isinstance(hyp, str)

    if isinstance(ref, str):
        ref = ref.split('||<|>||')  # special delimiter for backward compatibility
    ref = [a.strip() for a in ref]
    refs = {0: ref}
    ref_list = [ref]

    hyps = {0: [hyp.strip()]}
    hyp_list = [hyp]

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc
            else:
                ret_scores[method] = score

    if not no_skipthoughts:
        from nlgeval.skipthoughts import skipthoughts
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        model = skipthoughts.load_model()
        encoder = skipthoughts.Encoder(model)
        vector_hyps = encoder.encode([h.strip() for h in hyp_list], verbose=False)
        ref_list_T = np.array(ref_list).T.tolist()
        vector_refs = map(lambda refl: encoder.encode([r.strip() for r in refl], verbose=False), ref_list_T)
        cosine_similarity = list(map(lambda refv: cosine_similarity(refv, vector_hyps).diagonal(), vector_refs))
        cosine_similarity = np.max(cosine_similarity, axis=0).mean()
        ret_scores['SkipThoughtCS'] = cosine_similarity

    if not no_glove:
        from nlgeval.word2vec.evaluate import eval_emb_metrics
        import numpy as np

        glove_hyps = [h.strip() for h in hyp_list]
        ref_list_T = np.array(ref_list).T.tolist()
        glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
        scores = eval_emb_metrics(glove_hyps, glove_refs)
        scores = scores.split('\n')
        for score in scores:
            name, value = score.split(':')
            value = float(value.strip())
            ret_scores[name] = value

    return ret_scores


class NLGEval(object):
    def __init__(self, no_overlap=False, no_skipthoughts=False, no_glove=False):
        self.no_overlap = no_overlap
        if not no_overlap:
            self.load_scorers()

        self.no_skipthoughts = no_skipthoughts
        if not self.no_skipthoughts:
            self.load_skipthought_model()

        self.no_glove = no_glove
        if not self.no_glove:
            self.load_glove()

    def load_scorers(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

    def load_skipthought_model(self):
        from nlgeval.skipthoughts import skipthoughts
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        self.np = np
        self.cosine_similarity = cosine_similarity

        model = skipthoughts.load_model()
        self.skipthought_encoder = skipthoughts.Encoder(model)

    def load_glove(self):
        from nlgeval.word2vec.evaluate import Embedding
        from nlgeval.word2vec.evaluate import eval_emb_metrics
        import numpy as np
        self.eval_emb_metrics = eval_emb_metrics
        self.np = np
        self.glove_emb = Embedding()

    def compute_individual_metrics(self, ref, hyp):
        assert isinstance(hyp, str)
        ref = [a.strip() for a in ref]
        refs = {0: ref}
        ref_list = [ref]

        hyps = {0: [hyp.strip()]}
        hyp_list = [hyp]

        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, scs, m in zip(score, scores, method):
                        ret_scores[m] = sc
                else:
                    ret_scores[method] = score

        if not self.no_skipthoughts:
            vector_hyps = self.skipthought_encoder.encode([h.strip() for h in hyp_list], verbose=False)
            ref_list_T = self.np.array(ref_list).T.tolist()
            vector_refs = map(lambda refl: self.skipthought_encoder.encode([r.strip() for r in refl], verbose=False), ref_list_T)
            cosine_similarity = list(map(lambda refv: self.cosine_similarity(refv, vector_hyps).diagonal(), vector_refs))
            cosine_similarity = self.np.max(cosine_similarity, axis=0).mean()
            ret_scores['SkipThoughtCS'] = cosine_similarity

        if not self.no_glove:
            glove_hyps = [h.strip() for h in hyp_list]
            ref_list_T = self.np.array(ref_list).T.tolist()
            glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
            scores = self.eval_emb_metrics(glove_hyps, glove_refs, emb=self.glove_emb)
            scores = scores.split('\n')
            for score in scores:
                name, value = score.split(':')
                value = float(value.strip())
                ret_scores[name] = value

        return ret_scores

    def compute_metrics(self, ref_list, hyp_list):
        ref_list = [list(map(str.strip, refs)) for refs in zip(*ref_list)]
        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
        assert len(refs) == len(hyps)

        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, scs, m in zip(score, scores, method):
                        ret_scores[m] = sc
                else:
                    ret_scores[method] = score

        if not self.no_skipthoughts:
            vector_hyps = self.skipthought_encoder.encode([h.strip() for h in hyp_list], verbose=False)
            ref_list_T = self.np.array(ref_list).T.tolist()
            vector_refs = map(lambda refl: self.skipthought_encoder.encode([r.strip() for r in refl], verbose=False), ref_list_T)
            cosine_similarity = list(map(lambda refv: self.cosine_similarity(refv, vector_hyps).diagonal(), vector_refs))
            cosine_similarity = self.np.max(cosine_similarity, axis=0).mean()
            ret_scores['SkipThoughtCS'] = cosine_similarity

        if not self.no_glove:
            glove_hyps = [h.strip() for h in hyp_list]
            ref_list_T = self.np.array(ref_list).T.tolist()
            glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
            scores = self.eval_emb_metrics(glove_hyps, glove_refs, emb=self.glove_emb)
            scores = scores.split('\n')
            for score in scores:
                name, value = score.split(':')
                value = float(value.strip())
                ret_scores[name] = value

        return ret_scores
