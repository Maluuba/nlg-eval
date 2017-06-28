# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
from gensim.models import Word2Vec
from nlgeval.word2vec.glove2word2vec import glove2word2vec
import os

def txt2bin(filename):
    m = Word2Vec.load_word2vec_format(filename)
    m.vocab[m.vocab.keys()[0]].sample_int = 1
    m.save(filename.replace('txt','bin'), separately=None)
    Word2Vec.load(filename.replace('txt', 'bin'), mmap='r')


if __name__ == "__main__":

    path = os.path.join(os.path.dirname(__file__), "..", "data")
    glove_vector_file = os.path.join(path, 'glove.6B.300d.txt')
    output_model_file = os.path.join(path, 'glove.6B.300d.model.txt')

    txt2bin(glove2word2vec(glove_vector_file, output_model_file))
