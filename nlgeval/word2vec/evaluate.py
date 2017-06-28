# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
import os
import numpy as np
from gensim.models import Word2Vec


class Embedding(object):
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.m = Word2Vec.load(os.path.join(path, 'glove.6B.300d.model.bin'), mmap='r')
        self.unk = self.m.syn0.mean(axis=0)

    @property
    def w2v(self):
        return np.concatenate((self.m.syn0, self.unk[None,:]), axis=0)

    def __getitem__(self, key):
        try:
            return self.m.vocab[key].index
        except KeyError:
            return len(self.m.syn0)

    def vec(self, key):
        try:
            return self.m.syn0[self.m.vocab[key].index]
        except KeyError:
            return self.unk


def eval_emb_metrics(hypothesis, references):
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import word_tokenize
    import numpy as np
    emb = Embedding()

    emb_hyps = []
    avg_emb_hyps = []
    extreme_emb_hyps = []
    for hyp in hypothesis:
        embs = [emb.vec(word) for word in word_tokenize(hyp)]

        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        assert not np.any(np.isnan(avg_emb))

        maxemb = np.max(embs, axis=0)
        minemb = np.min(embs, axis=0)
        extreme_emb = map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb)

        emb_hyps.append(embs)
        avg_emb_hyps.append(avg_emb)
        extreme_emb_hyps.append(extreme_emb)

    emb_refs = []
    avg_emb_refs = []
    extreme_emb_refs = []
    for refsource in references:
        emb_refsource = []
        avg_emb_refsource = []
        extreme_emb_refsource = []
        for ref in refsource:
            embs = [emb.vec(word) for word in word_tokenize(ref)]

            avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
            assert not np.any(np.isnan(avg_emb))

            maxemb = np.max(embs, axis=0)
            minemb = np.min(embs, axis=0)
            extreme_emb = map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb)

            emb_refsource.append(embs)
            avg_emb_refsource.append(avg_emb)
            extreme_emb_refsource.append(extreme_emb)
        emb_refs.append(emb_refsource)
        avg_emb_refs.append(avg_emb_refsource)
        extreme_emb_refs.append(extreme_emb_refsource)

    cos_similarity = map(lambda refv: cosine_similarity(refv, avg_emb_hyps).diagonal(), avg_emb_refs)
    cos_similarity = np.max(cos_similarity, axis=0).mean()
    average = "EmbeddingAverageCosineSimilairty: %0.6f" % (cos_similarity)

    cos_similarity = map(lambda refv: cosine_similarity(refv, extreme_emb_hyps).diagonal(), extreme_emb_refs)
    cos_similarity = np.max(cos_similarity, axis=0).mean()
    extrema = "VectorExtremaCosineSimilarity: %0.6f" % (cos_similarity)

    scores = []
    for emb_refsource in emb_refs:
        score_source = []
        for emb_ref, emb_hyp in zip(emb_refsource, emb_hyps):
            simi_matrix = cosine_similarity(emb_ref, emb_hyp)
            dir1 = simi_matrix.max(axis=0).mean()
            dir2 = simi_matrix.max(axis=1).mean()
            score_source.append((dir1+dir2)/2)
        scores.append(score_source)
    scores = np.max(scores, axis=0).mean()
    greedy = "GreedyMatchingScore: %0.6f" % (scores)

    rval = "\n".join([average, extrema, greedy])
    return rval


if __name__ == '__main__':
    emb = Embedding()
