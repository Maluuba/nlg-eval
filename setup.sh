#!/bin/bash

set -e

function download () {
	URL=$1
	TGTDIR=.
	if [ -n "$2" ]; then
		TGTDIR=$2
		mkdir -p $TGTDIR
	fi
	wget $URL -P $TGTDIR
}

function download_file() {
	URL=$1
	filename="${URL##*/}"
	TGTDIR=.
	if [ -n "$2" ]; then
		TGTDIR=$2
	fi
	if [ ! -f "$TGTDIR/$filename" ]; then
		download $@
	fi
}

# glove data
if [ ! -f nlgeval/data/glove.6B.300d.txt ]
then
    download http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    rm -f glove.6B.50d.txt
    rm -f glove.6B.100d.txt
    rm -f glove.6B.200d.txt
    rm -f glove.6B.zip
    mkdir -p nlgeval/data
    mv glove.6B.300d.txt nlgeval/data
fi

# skip-thoughts data
download_file http://www.cs.toronto.edu/~rkiros/models/dictionary.txt nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/utable.npy nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/btable.npy nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz nlgeval/data
download_file http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl nlgeval/data

# multi-bleu.perl
download_file https://raw.githubusercontent.com/moses-smt/mosesdecoder/b199e654df2a26ea58f234cbb642e89d9c1f269d/scripts/generic/multi-bleu.perl nlgeval/multibleu
[ -e nlgeval/multibleu/multi-bleu.perl ] && chmod +x nlgeval/multibleu/multi-bleu.perl

# glove word vectors
download_file https://raw.githubusercontent.com/manasRK/glove-gensim/42ce46f00e83d3afa028fb6bf17ed3c90ca65fcc/glove2word2vec.py nlgeval/word2vec

if [ ! -f nlgeval/data/glove.6B.300d.model.bin ]
then
    python2.7 -m nltk.downloader punkt
    PYTHONPATH=`pwd` python2.7 nlgeval/word2vec/generate_w2v_files.py
fi
